import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from src.training import util as sd
from src.training.util import DATASET_DIR
from src.branch_3D import UniMolModel
from src.training.CustomizedDataset_GGT_InM import MyOwnDataset
from src.training.GGTmodel_3 import Net


# =============================================================================
# Result directories / Slurm info
# =============================================================================

os.makedirs("results", exist_ok=True)

job_id = os.environ.get("SLURM_JOB_ID", "nojid")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "notask")


# =============================================================================
# Model builder
# =============================================================================

def build_model_from_args(args, number_of_task, device):
    # resolve "none" string from argparse to actual None for single modality
    fusion = None if args.fuse_mechanism == "none" else args.fuse_mechanism

    print("========== Model Config ==========")
    print("modality :", args.enabled_modality)
    print("fusion   :", fusion)
    print("==================================")

    model = Net(
        n_output_layers=number_of_task,
        modality=args.enabled_modality,   # ← was missing
        fusion=fusion,
    ).to(device)

    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def model_eval(args, model, device, loader, num_sample, tokenizer, criterion):
    """
    Evaluate model on a data loader.
    """
    number_of_task = args.n_tasks
    eval_loss = 0.0
    preds = []
    labels = []

    model.eval()

    for data in tqdm(loader):
        smiles = data["graph"].smiles

        if args.token_length_smile == 0:
            inputs = tokenizer.batch_encode_plus(
                smiles,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer.batch_encode_plus(
                smiles,
                max_length=args.token_length_smile,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        inputs = inputs.to(device)
        graph = data["graph"].to(device)
        unimol_input = {k: v.to(device) for k, v in data["unimol_input"].items()}

        outputs = model(
            graph=graph,
            inputs=inputs,
            unimol_input=unimol_input,
        )

        y = torch.as_tensor(
            data["target"],
            device=device,
            dtype=torch.float32,
        )

        if outputs.ndim == 2 and outputs.size(1) == 1 and y.ndim == 1:
            y = y.unsqueeze(1)

        eval_loss += criterion(outputs.float(), y).item()

        preds += torch.sigmoid(outputs).view(-1).detach().cpu().tolist()
        labels += y.detach().cpu().tolist()

    trues = np.array(labels).reshape(-1, number_of_task).T
    belief_scores = np.array(preds).reshape(-1, number_of_task).T

    roc_auc_score_list = [
        roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
        for i in range(number_of_task)
    ]

    auc_roc = float(sum(roc_auc_score_list) / number_of_task)
    accuracy = accuracy_score(labels, [1 if v > 0.5 else 0 for v in preds])

    eval_loss = eval_loss / num_sample

    return eval_loss, accuracy, auc_roc


# =============================================================================
# Optimizer / Scheduler factory
# =============================================================================

def build_optimizer_and_scheduler(model, args, T_max=50):
    """
    Build AdamW optimizer and optional LR scheduler.

    Only parameters with requires_grad=True are passed into optimizer.
    This is important when using model.set_train_stage("fusion_only").

    Supported schedulers:
        - none / constant: no scheduler
        - cosine: CosineAnnealingLR
        - plateau: ReduceLROnPlateau
        - step: StepLR
    """

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler_name = getattr(args, "scheduler", "cosine")

    # ============================================================
    # 1. No scheduler / constant LR
    # ============================================================
    if scheduler_name in ["none", "constant"]:
        print("Scheduler: none / constant LR")
        scheduler = None
        return optimizer, scheduler

    # ============================================================
    # 2. CosineAnnealingLR
    # ============================================================
    if scheduler_name == "cosine":
        # Priority:
        # 1. args.t_max, if provided and positive
        # 2. function argument T_max, if provided and positive
        # 3. args.epochs
        if hasattr(args, "t_max") and args.t_max is not None and args.t_max > 0:
            final_T_max = args.t_max
        elif T_max is not None and T_max > 0:
            final_T_max = T_max
        else:
            final_T_max = args.epochs

        print("Scheduler: cosine")
        print("T_max is", final_T_max)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, final_T_max),
        )

        return optimizer, scheduler

    # ============================================================
    # 3. ReduceLROnPlateau
    # ============================================================
    if scheduler_name == "plateau":
        print("Scheduler: ReduceLROnPlateau")
        print("plateau_factor:", args.plateau_factor)
        print("plateau_patience:", args.plateau_patience)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
        )

        return optimizer, scheduler

    # ============================================================
    # 4. StepLR
    # ============================================================
    if scheduler_name == "step":
        print("Scheduler: StepLR")
        print("step_size:", args.step_size)
        print("gamma:", args.gamma)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )

        return optimizer, scheduler

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = sd.parse_input()

    sd.set_global_seed(args.random_seed)

    epochs = args.epochs
    number_of_task = args.n_tasks
    fusion_end_epochs = args.freeze_epoch

    cv_enabled = getattr(args, "cv_folds", 0) is not None and args.cv_folds > 1
    cv_valid_fold_id = None

    run_name = args.model_name
    if cv_enabled:
        if args.cv_fold_id < 0 or args.cv_fold_id >= args.cv_folds:
            raise ValueError(f"cv_fold_id must be in [0, {args.cv_folds - 1}], got {args.cv_fold_id}")
        cv_valid_fold_id = (args.cv_fold_id + args.cv_valid_fold_offset) % args.cv_folds
        if cv_valid_fold_id == args.cv_fold_id:
            raise ValueError("cv validation fold must be different from test fold. Change --cv_valid_fold_offset.")
        run_name = f"{args.model_name}_cv{args.cv_folds}_testfold{args.cv_fold_id}_validfold{cv_valid_fold_id}"

    # ── TensorBoard ──────────────────────────────────────────────────────────
    tensorboard_log_dir = os.path.join(args.tensorboard_dir, run_name)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    print(f"[TensorBoard] {tensorboard_log_dir}")

    # ── Device & tokenizer ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # ── Dataset ──────────────────────────────────────────────────────────────
    molecular_dataset = MyOwnDataset(
        str(DATASET_DIR / args.dataset),
        dataset_name=args.dataset,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model_from_args(
        args=args,
        number_of_task=number_of_task,
        device=device,
    )

    # ── Initialize staged training ───────────────────────────────────────────
    if fusion_end_epochs > 0:
        current_stage = "fusion_only"
    else:
        current_stage = "unfreeze_all"

    model.set_train_stage(current_stage)

    print(f"[Train stage init] {current_stage}")
    model_batch = UniMolModel(
        output_dim=767,
        data_type="molecule",
        remove_hs=False,
    )

    # ── Data split ───────────────────────────────────────────────────────────
    is_balance = False

    if cv_enabled:
        print("---- in scaffold K-fold cross-validation")

        (
            train_loader,
            valid_loader,
            test_loader,
            train_size,
            valid_size,
            test_size,
        ) = sd.scaffold_k_fold_split(
            dataset=molecular_dataset,
            k=args.cv_folds,
            fold_id=args.cv_fold_id,
            valid_fold_id=cv_valid_fold_id,
            seed=args.random_seed,
            batch_size=args.batch_size,
            collate_fn=model_batch.batch_collate_fn_2,
            save_split=True,
            split_name=(
                f"{args.dataset}_seed{args.random_seed}_"
                f"cv{args.cv_folds}_testfold{args.cv_fold_id}_validfold{cv_valid_fold_id}"
            ),
        )

    elif args.random_scaffold:
        print("---- in random scaffold")

        (
            train_loader,
            valid_loader,
            test_loader,
            train_size,
            valid_size,
            test_size,
        ) = sd.random_scaffold_split(
            dataset=molecular_dataset,
            null_value=0,
            smiles_list=[],
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.random_seed,
            batch_size=args.batch_size,
            collate_fn=model_batch.batch_collate_fn_2,
        )
    else:
        (
            train_loader,
            valid_loader,
            test_loader,
            train_size,
            valid_size,
            test_size,
        ) = sd.split_data(
            molecular_dataset,
            args.random_seed,
            args.batch_size,
            is_balance,
        )

    total_data = train_size + valid_size + test_size
    print("The length of data:", total_data)
    print("train_size, valid_size, test_size:", train_size, valid_size, test_size)

    print(
        "len(train_loader), len(valid_loader), len(test_loader):",
        len(train_loader),
        len(valid_loader),
        len(test_loader),
    )

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = torch.nn.BCEWithLogitsLoss()

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        args=args,
        T_max=epochs,
    )

    # ── Metric history ───────────────────────────────────────────────────────
    train_loss_list = []
    train_accuracy_list = []
    train_auc_roc_list = []

    valid_loss_list = []
    valid_accuracy_list = []
    valid_auc_roc_list = []

    test_loss_list = []
    test_accuracy_list = []
    test_auc_roc_list = []

    current_lr_list = []
    lr_after_step_list = []

    # ── Track best model on disk, not in CPU memory ──────────────────────────
    best_valid_auc = -1.0
    best_epoch = 0

    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"{checkpoint_dir}/3d_branch_{args.dataset}_{run_name}.pth"

    # =============================================================================
    # Training loop
    # =============================================================================

    for epoch in range(epochs):
        # ── Switch from fusion_only to unfreeze_all at freeze_epoch ───────────
        if epoch == fusion_end_epochs and fusion_end_epochs > 0:
            current_stage = "unfreeze_all"
            model.set_train_stage(current_stage)

            remaining = epochs - fusion_end_epochs

            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                args=args,
                T_max=remaining,
            )

            print(f"[Train stage switch] epoch={epoch}, stage={current_stage}")

        current_lr = optimizer.param_groups[0]["lr"]
        current_lr_list.append(current_lr)

        writer.add_scalar("LR/current_lr", current_lr, epoch)
        print(f"[Epoch {epoch}] current_lr = {current_lr:.8g}")

        start_time = time.time()

        # model.train() sets all submodules to train mode,
        # so we re-apply set_train_stage() after it.
        model.train()
        model.set_train_stage(current_stage)

        print(f"[Epoch {epoch}] current_stage = {current_stage}")

        train_loss_sum = 0.0
        preds = []
        labels = []

        for data in tqdm(train_loader):
            smiles = data["graph"].smiles

            if args.token_length_smile == 0:
                inputs = tokenizer.batch_encode_plus(
                    smiles,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer.batch_encode_plus(
                    smiles,
                    max_length=args.token_length_smile,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

            inputs = inputs.to(device)
            graph = data["graph"].to(device)
            unimol_input = {k: v.to(device) for k, v in data["unimol_input"].items()}

            outputs = model(
                graph=graph,
                inputs=inputs,
                unimol_input=unimol_input,
            )

            y = torch.as_tensor(
                data["target"],
                device=device,
                dtype=torch.float32,
            )

            if outputs.ndim == 2 and outputs.size(1) == 1 and y.ndim == 1:
                y = y.unsqueeze(1)

            loss = criterion(outputs.float(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            preds += torch.sigmoid(outputs).view(-1).detach().cpu().tolist()
            labels += y.detach().cpu().tolist()


        # ── Train metrics ────────────────────────────────────────────────────
        train_loss = train_loss_sum / train_size
        print("Train/Loss", train_loss, epoch)

        trues = np.array(labels).reshape(-1, number_of_task).T
        belief_scores = np.array(preds).reshape(-1, number_of_task).T

        roc_auc_score_list = [
            roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
            for i in range(number_of_task)
        ]

        train_auc_roc = float(sum(roc_auc_score_list) / number_of_task)
        train_accuracy = accuracy_score(
            labels,
            [1 if v > 0.5 else 0 for v in preds],
        )

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_accuracy, epoch)
        writer.add_scalar("Train/AUC-ROC", train_auc_roc, epoch)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        train_auc_roc_list.append(train_auc_roc)

        # ── Validation & test ────────────────────────────────────────────────
        valid_loss, valid_accuracy, valid_auc_roc = model_eval(
            args=args,
            model=model,
            device=device,
            loader=valid_loader,
            num_sample=valid_size,
            tokenizer=tokenizer,
            criterion=criterion,
        )

        test_loss, test_accuracy, test_auc_roc = model_eval(
            args=args,
            model=model,
            device=device,
            loader=test_loader,
            num_sample=test_size,
            tokenizer=tokenizer,
            criterion=criterion,
        )

        # ── LR scheduler update ───────────────────────────────────────────────
        if scheduler is not None:
            if args.scheduler == "plateau":
                # ReduceLROnPlateau needs the validation metric.
                # mode="max", so we pass valid_auc_roc.
                scheduler.step(valid_auc_roc)
            else:
                # CosineAnnealingLR and StepLR do not need validation metric.
                scheduler.step()

        lr_after_step = optimizer.param_groups[0]["lr"]
        lr_after_step_list.append(lr_after_step)

        writer.add_scalar("LR/lr_after_step", lr_after_step, epoch)
        print(f"[Epoch {epoch}] lr_after_step = {lr_after_step:.8g}")

        writer.add_scalar("Valid/Loss", valid_loss, epoch)
        writer.add_scalar("Valid/Acc", valid_accuracy, epoch)
        writer.add_scalar("Valid/AUC_ROC", valid_auc_roc, epoch)

        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc", test_accuracy, epoch)
        writer.add_scalar("Test/AUC_ROC", test_auc_roc, epoch)

        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)
        valid_auc_roc_list.append(valid_auc_roc)

        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        test_auc_roc_list.append(test_auc_roc)

        # # ── Snapshot best model weights in memory ────────────────────────────
        # if valid_auc_roc > best_valid_auc:
        #     best_valid_auc = valid_auc_roc
        #     best_epoch = epoch
        #
        #     torch.save(
        #         {
        #             "model_state": model.state_dict(),
        #             "optimizer_state": optimizer.state_dict(),
        #             "epoch": best_epoch,
        #             "valid_auc": best_valid_auc,
        #         },
        #         checkpoint_name,
        #     )
        #
        #     print(
        #         f"[Best checkpoint saved] epoch={best_epoch}, "
        #         f"valid_auc={best_valid_auc:.4f}, path={checkpoint_name}"
        #     )

        stop_time = time.time()
        print("time is:{:.4f}s".format(stop_time - start_time))

    # =============================================================================
    # Post-training: best epoch by validation AUC
    # =============================================================================

    max_index_in_valid = valid_auc_roc_list.index(max(valid_auc_roc_list))

    best_valid_accuracy = valid_accuracy_list[max_index_in_valid]
    best_valid_auc_roc = valid_auc_roc_list[max_index_in_valid]
    best_test_accuracy = test_accuracy_list[max_index_in_valid]
    best_test_auc_roc = test_auc_roc_list[max_index_in_valid]

    # ── CSV export: compact best metrics file ────────────────────────────────
    df = pd.DataFrame(
        {
            "Seed": [args.random_seed],
            "Best Validation Accuracy": [best_valid_accuracy],
            "Best Validation AUC ROC": [best_valid_auc_roc],
            "Best Test Accuracy": [best_test_accuracy],
            "Best Test AUC ROC": [best_test_auc_roc],
        }
    ).set_index("Seed")

    csv_file = args.key + "_key_" + args.dataset + "_best_metrics.csv"

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, index_col="Seed")
        combined_df = pd.concat([existing_df, df])
        combined_df.to_csv(csv_file)
    else:
        df.to_csv(csv_file)

    # =============================================================================
    # Added function 1:
    # Save train / valid / test metrics for every epoch
    # =============================================================================

    epoch_df = pd.DataFrame(
        {
            "epoch": list(range(epochs)),

            "current_lr": current_lr_list,
            "lr_after_step": lr_after_step_list,

            "train_loss": train_loss_list,
            "train_acc": train_accuracy_list,
            "train_auc_roc": train_auc_roc_list,

            "valid_loss": valid_loss_list,
            "valid_acc": valid_accuracy_list,
            "valid_auc_roc": valid_auc_roc_list,

            "test_loss": test_loss_list,
            "test_acc": test_accuracy_list,
            "test_auc_roc": test_auc_roc_list,
        }
    )

    os.makedirs(args.csv_dir, exist_ok=True)

    csv_name = f"{run_name}.job{job_id}_task{task_id}.epoch_metrics.csv"
    csv_path = os.path.join(args.csv_dir, csv_name)

    epoch_df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")



    print(
        f"Checkpoint saved: best epoch {best_epoch}, "
        f"valid AUC={best_valid_auc_roc:.4f}, "
        f"test AUC={best_test_auc_roc:.4f}"
    )

    writer.close()

    # =============================================================================
    # Added function 2:
    # Store best result selected by validation AUC into one master CSV
    # =============================================================================

    best_epoch = int(np.argmax(valid_auc_roc_list))

    best_row = {
        # identifiers
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "model_name": getattr(args, "model_name", "no_model_name"),
        "run_name": run_name,
        "key": args.key,

        # cross-validation info
        "cv_enabled": bool(cv_enabled),
        "cv_folds": int(args.cv_folds) if cv_enabled else "",
        "cv_test_fold_id": int(args.cv_fold_id) if cv_enabled else "",
        "cv_valid_fold_id": int(cv_valid_fold_id) if cv_enabled else "",
        "seed": args.random_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,

        # scheduler info
        "scheduler": str(getattr(args, "scheduler", "")),
        "t_max": int(getattr(args, "t_max", 0)) if getattr(args, "t_max", None) is not None else "",
        "step_size": int(getattr(args, "step_size", 0)) if getattr(args, "step_size", None) is not None else "",
        "gamma": float(getattr(args, "gamma", 0.0)) if getattr(args, "gamma", None) is not None else "",
        "plateau_patience": int(getattr(args, "plateau_patience", 0)) if getattr(args, "plateau_patience", None) is not None else "",
        "plateau_factor": float(getattr(args, "plateau_factor", 0.0)) if getattr(args, "plateau_factor", None) is not None else "",

        "freeze_epoch": args.freeze_epoch,
        "initial_train_stage": "freeze_1d_3d" if args.freeze_epoch > 0 else "unfreeze_all",
        "unfreeze_stage": "unfreeze_all",

        "enabled_modality": " ".join(args.enabled_modality)
        if isinstance(args.enabled_modality, (list, tuple))
        else str(args.enabled_modality),

        "fuse_mechanism": str(args.fuse_mechanism),
        "strategy": str(getattr(args, "strategy", "")),

        # Slurm ids
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),

        # selection info
        "best_epoch_by": "valid_auc_roc",
        "best_epoch": best_epoch,

        # train metrics at best validation epoch
        "train_loss": float(train_loss_list[best_epoch]),
        "train_acc": float(train_accuracy_list[best_epoch]),
        "train_auc_roc": float(train_auc_roc_list[best_epoch]),

        # validation metrics at best validation epoch
        "valid_loss": float(valid_loss_list[best_epoch]),
        "valid_acc": float(valid_accuracy_list[best_epoch]),
        "valid_auc_roc": float(valid_auc_roc_list[best_epoch]),

        # test metrics at best validation epoch
        "test_loss": float(test_loss_list[best_epoch]),
        "test_acc": float(test_accuracy_list[best_epoch]),
        "test_auc_roc": float(test_auc_roc_list[best_epoch]),
    }

    best_df = pd.DataFrame([best_row])

    job_dir = f"job{job_id}"

    master_csv_dir = os.path.join(
        "results_bestvalidation",
        args.dataset,
        job_dir,
    )

    os.makedirs(master_csv_dir, exist_ok=True)

    master_csv = os.path.join(
        master_csv_dir,
        f"{args.dataset}_{args.key}_MASTER_best_by_valid_auc.csv",
    )

    if os.path.exists(master_csv):
        best_df.to_csv(master_csv, mode="a", header=False, index=False)
    else:
        best_df.to_csv(master_csv, mode="w", header=True, index=False)

    print(f"[Saved/Append] {master_csv}")
    print(
        f"[Best epoch] {best_epoch} | "
        f"valid_auc={best_row['valid_auc_roc']:.4f} | "
        f"test_auc={best_row['test_auc_roc']:.4f}"
    )