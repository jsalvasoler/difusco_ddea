from config.myconfig import Config

config = Config(
    task="mis",
    diffusion_type="gaussian",
    learning_rate=0.0002,
    weight_decay=0.0001,
    lr_scheduler="cosine-decay",
    diffusion_steps=1000,
    validation_examples=8,
    diffusion_schedule="linear",
    inference_schedule="cosine",
    inference_diffusion_steps=50,
    device="cuda",
    inference_trick="ddim",
    sparse_factor=-1,
    n_layers=12,
    hidden_dim=256,
    aggregation="sum",
    use_activation_checkpoint=True,
    fp16=False,
)
