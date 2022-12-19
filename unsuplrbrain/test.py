import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import SparseAutoencoder

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.25))
        ])
CKPT_PATH = '//home/chandramouli/kaggle/project/logs/left-right-brain/layer=1|lr=1.0e-4|wd=1.0e-5|bs=32|opt=adam|k=3|%k=0.3|/checkpoints/last.ckpt'

# 1.0e-4 acc = 2.375%
# 1.0e-3 acc = 1.875%
# 
TEST_FOLDER = '/kaggle/project/omniglot/python/images_evaluation/Manipuri'
dataset = ImageFolder(root=TEST_FOLDER,
                        transform=test_transforms,
                        loader=pil_loader,)

dataloader = DataLoader(dataset, 
                            batch_size=20, 
                            num_workers=4)

model = SparseAutoencoder().to(device)
checkpoint = torch.load(CKPT_PATH)
checkpoint['state_dict'] = {k.replace('model.',''):v \
                for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(checkpoint['state_dict'])
model.eval()
targets = []
outputs = []
for (ind, batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
    img, target = batch
    img = img.to(device)
    out = model(img, training=False)
    out = out[0].detach().cpu()
    out = out.reshape(target.shape[0], -1)
    targets.append(target)
    outputs.append(out)

targets = torch.stack(targets)
outputs = torch.stack(outputs)
outputs = outputs.view(-1, outputs.shape[-1]).numpy()
targets = targets.view(-1).numpy()

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
kmeans = KMeans(n_clusters=40, random_state=42)
labels = kmeans.fit_predict(outputs)
acc1 = accuracy_score(targets, labels)

TEST_FOLDER = '/home/chandramouli/kaggle/project/omniglot/python/images_evaluation/Atlantean'
dataset = ImageFolder(root=TEST_FOLDER,
                        transform=test_transforms,
                        loader=pil_loader,)

dataloader = DataLoader(dataset, 
                            batch_size=20, 
                            num_workers=4)

targets = []
outputs = []
for (ind, batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
    img, target = batch
    img = img.to(device)
    out = model(img, training=False)
    out = out[0].detach().cpu()
    out = out.reshape(target.shape[0], -1)
    targets.append(target)
    outputs.append(out)

targets = torch.stack(targets)
outputs = torch.stack(outputs)
outputs = outputs.view(-1, outputs.shape[-1]).numpy()
targets = targets.view(-1).numpy()

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
kmeans = KMeans(n_clusters=40, random_state=42)
labels = kmeans.fit_predict(outputs)
acc2 = accuracy_score(targets, labels)
# clf = LogisticRegression(random_state=0, max_iter=500, 
#             solver='liblinear').fit(outputs, targets)
# labels = clf.predict(outputs)
# acc = accuracy_score(targets, labels)
# print(acc)
print(f'Accuracy on Manipuri test dataset : {acc1}\nAccuracy on Atlantean test dataset : {acc2}')