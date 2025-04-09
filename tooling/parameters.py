import sys
from pathlib import Path

import torch
from prettytable import PrettyTable

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.model14 import DocumentSeparator, ImageEncoder, TextEncoder
from utils.text_utils import combine_texts


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


_input = {
    "images": torch.randn(1, 3, 3, 512, 512),
    "texts": [[{"a": {"text": "lorem ipsum dolor sit amet, consectetur adipiscing elit." * 80}}] * 3],
    "shapes": torch.randn(1, 3, 2),
}

net = DocumentSeparator.load_from_checkpoint(
    "/home/stefan/Downloads/version_4/checkpoints/document_separator-epoch=31-val_acc=0.9261.ckpt"
)
del net.fc_middle
del net.fc_end
del net.image_encoder.conv2d
net(_input)

# net = TextEncoder(merge_to_batch=True).to("cuda")
# texts = _input["texts"]
# for i in range(len(texts)):
#     for j in range(len(texts[i])):
#         texts[i][j] = combine_texts(x["text"] for x in texts[i][j].values())
# net(texts)


# net = ImageEncoder(merge_to_batch=True).to("cuda")
# net(_input["images"])

count_parameters(net)
