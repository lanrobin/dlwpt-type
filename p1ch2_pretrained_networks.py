import torch
from torchvision import models
from torchvision import transforms

from PIL import Image

from common_utils import DATA_ROOT
from common_utils import get_files_with_extensions

alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)

#resnet

preprocess = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

resnet.eval()

images = get_files_with_extensions(f"{DATA_ROOT}/p1ch2", ['jpg', 'webp'])
labels = []
with open(f"{DATA_ROOT}/p1ch2/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

for img_name in images:
    print(f"-----Eval image:{img_name}----")
    img = Image.open(f"{DATA_ROOT}/p1ch2/{img_name}")
    img_t = preprocess(img)

    batch_t = torch.unsqueeze(img_t, 0)



    out = resnet(batch_t)

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    #(labels[index[0]],percentage[index[0]].item())
    print(f"MaxPossibility:{str((img_name, labels[index[0]],percentage[index[0]].item()))}")

    _, indices = torch.sort(out, descending=True)
    [print(str((img_name, labels[idx], percentage[idx].item()))) for idx in indices[0][:5]]