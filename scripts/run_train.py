import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from human_ai.code.dataset import get_train_dataset
from human_ai.code.model import RobertaHumanAI

dataset = get_train_dataset()
training_args = dict(
    output_dir="/shared/experiments/lymph_nodes/zones/zones_az/debug/exp0",
    per_device_train_batch_size=28,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="epoch",
    num_train_epochs=100,
    learning_rate=1e-4,
    report_to="wandb"
)

model = RobertaHumanAI(training_args)
model.train(dataset)
