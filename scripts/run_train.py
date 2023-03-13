import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from human_ai.code.dataset import get_train_dataset
from human_ai.code.model import ClassificatorHumanAI

dataset = get_train_dataset(test_size=0.15)

# model_name = "roberta-base-openai-detector"
# training_args = dict(
#     output_dir="/shared/experiments/lymph_nodes/zones/zones_az/debug/exp0",
#     per_device_train_batch_size=28,
#     evaluation_strategy="steps",
#     eval_steps=100,
#     logging_strategy="epoch",
#     num_train_epochs=100,
#     learning_rate=1e-4,
#     report_to="wandb"
# )


training_args = dict(
    output_dir="/shared/experiments/lymph_nodes/zones/zones_az/debug/exp1_bs32",
    per_device_train_batch_size=32,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=60,
    learning_rate=1e-5,
    report_to="wandb"
)

model_name = "distilbert-base-uncased"
model = ClassificatorHumanAI(training_args, model_name)
model.train(dataset)
