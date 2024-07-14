# Scalable Application

Scalable applications refers to accelerating data loading and model training by utilizing parallalization with both CPU and GPU. Moreover, we are also interested in conducting scalable inference, i.e., decreasing the inference time of our model if a user sends a request with methods like model pruning, quantization and model distillation.

## Distributed Data Loading

Distributed data loading is achieved by utilizing multiple cores/threads of a CPU to handle multiple processes in parallel. In our setup, we create the dataloaders such that we use 50% of the available workers. Additionally, we set `pin_memory=True` to store our dataset into GPU memory since our dataset is relatively small. By setting this flag we are telling Pytorch to lock the data in place in memory which will make the transfer between the host (CPU) and the device (GPU) faster. Additionaly, we set `persistent_workers=True`. The worker processes created by the DataLoader will remain active after each epoch ends, instead of being shut down and restarted for each new epoch. This can save time by avoiding the overhead of spawning new processes at the start of each epoch. Keeping the worker processes alive can lead to a more efficient data loading process, especially when training for many epochs.

```python
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=batch_size,
    num_workers=get_num_workers(),
    persistent_workers=True,
    pin_memory=True,  # for speeding up GPU utilization
    )
```

## Distributed Model Training

PyTorch Lightning naturally supports distributed data loading and model training. We can simply set the number of avaialbe devices (GPUs) in the Trainer. The argument `devices=cfg.train.devices` provides a flexible way to set the number of available GPUs with hydra.

```python
trainer = Trainer(
    profiler=cfg.train.profiler,
    precision=cfg.train.precision,
    max_epochs=cfg.train.epochs,
    callbacks=callbacks,
    accelerator=accelerator,
    devices=cfg.train.devices,
    log_every_n_steps=cfg.train.log_every_n_steps,
    enable_checkpointing=True,
    enable_model_summary=True,
    logger=wandb_logger,
    default_root_dir=model_dir,
    )
```

We also set:

```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```
As we instantiate parallel data loading before initializing the tokenizers, we have to set the parallelism feature for tokenizers to false since there is a risk to encounter a deadlock.

## Scalable Inference

In scalable inference, we aim to speed up the inference of our model. For that we can make use of multiple approaches such as quantization, pruning, quantization and model distillation.
A general approach to check the efficiency of a model is to examine the number of parameters and FLOPS (floating point operations) using the Thop package.

````python
flops, params = profile(plain_model, inputs=(input_ids, attention_mask))
````

We can also investigate the average inference time over a specified number of observations (for example 100).

### Quantization

Quantization is a technique where all computations are performed with integers instead of floats. We are essentially taking all continuous signals and converting them into discretized signals. This can increase inference time since integer operations are generall faster than float-point computations.

```python
torch.backends.quantized.engine = "qnnpack"
quantized_model = torch.quantization.quantize_dynamic(plain_model, {torch.nn.Linear}, dtype=torch.qint8)
```
qint8 is a dynamic quantization method that supports embeddings, which is required for BERT-based models. However, quantization doesn't necessairly need to increase inferene speed.

### Pruning

Pruning is a way for reducing model size and potentially improve performance. Pruning simply removes weights (setting to 0) that are not consider important for the task at hand. There are many ways to determine if a weight is important but the general rule that the importance of a weight is proportional to the magnitude of a given weight. This makes intuitively sense, since weights in all linear operations (fully connected or convolutional) are always multiplied onto the incoming value, thus a small weight means a small outgoing activation.

PyTorch Lightning naturally supports pruning:

```python
if cfg.train.pruning:
    pruning_callback = ModelPruning("l1_unstructured", amount=cfg.train.pruning_rate)
    callbacks.append(pruning_callback)
```

### Distillation

Knowledge distillation is somewhat similar to pruning, in the sense that it tries to find a smaller model that can perform equally well as a large model, however it does so in a completely different way. Knowledge distillation is a model compression technique that builds on the work of Bucila et al. in which we try do distill/compress the knowledge of a large complex model (also called the teacher model) into a simpler model (also called the student model).

For this we define our best finetuned model as the teacher model. We additionally define a smaller student model, i.e., considering a smaller layer size and smaller pretrained model. The larger finetuned model and the smaller student model are both required for the `distillation_loss`function. This loss function is used in the `training_step` of the `DistillationTrainer` class.

```python
def training_step(self, batch, batch_idx):
    """
    Defines a single step in the training loop.

    Args:
        batch (tuple): The input batch of data.
        batch_idx (int): The index of the batch.

    Returns:
        torch.Tensor: The computed loss for the batch.
    """
    sent_id, mask, labels = batch
    with torch.no_grad():
        teacher_outputs = self.teacher_model(sent_id, mask)

    student_outputs = self.student_model(sent_id, mask)
    loss = distillation_loss(student_outputs, labels, teacher_outputs, self.T, self.alpha)
    acc = (labels == student_outputs.argmax(dim=-1)).float().mean()

    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
```

```python
def distillation_loss(student_outputs, labels, teacher_outputs, temperature, alpha):
    """
    Computes the distillation loss.

    Args:
        student_outputs (torch.Tensor): Student model predictions.
        labels (torch.Tensor): Ground truth labels.
        teacher_outputs (torch.Tensor): Teacher model predictions (logits).
        temperature (float): Temperature parameter.
        alpha (float): Weight for distillation loss.

    Returns:
        torch.Tensor: Loss value combining cross-entropy and distillation loss.
    """
    hard_loss = F.cross_entropy(student_outputs, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction="batchmean",
    )
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

- **Hard Loss**: Standard cross-entropy loss between the student model's predictions and the true labels.
- **Soft Loss**: Kullback-Leibler (KL) divergence between the softened student outputs and softened teacher outputs.
  - **Temperature (T)**: The temperature parameter smooths the probability distributions, allowing the student model to learn from the teacher's "softer" predictions.

Parameters and their roles:

1. **Temperature (T)**
    - **Smoothing Factor**: Higher values of `T` produce softer probability distributions. It helps the student model learn not just the final prediction, but also the uncertainties of the teacher model.
    - **Effect on Soft Loss**: A higher temperature increases the contribution of less confident predictions, providing more informative gradients.

2. **Alpha (α)**
    - **Balance Factor**: Determines the weight of the soft loss (knowledge from teacher) versus the hard loss (true labels).
    - **Examples**:
        - \( \alpha = 0 \): Only true label loss is used (standard training).
        - \( \alpha = 1 \): Only soft label loss is used (pure distillation).

The **soft probability distributions** refer to the output probabilities from the teacher model that are smoothed using the temperature parameter. For example, if the teacher model is highly confident about its prediction, it might output a probability distribution like [0.9, 0.1]. When the temperature is increased, this distribution becomes softer, such as [0.6, 0.4]. This softening helps the student model learn from the nuances of the teacher model's predictions, including the uncertainties, rather than just the hard classifications.

Training Process:

During training, the student model learns from both the true labels and the teacher's softened predictions. The training step computes a combined loss from the teacher's guidance and true labels, guiding the student model's learning process.
