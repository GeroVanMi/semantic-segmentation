from torch import rand, randint
from torchmetrics import JaccardIndex

BATCH_SIZE = 3
target = randint(low=0, high=3, size=(BATCH_SIZE, 255, 255))
preds = rand(size=(BATCH_SIZE, 3, 255, 255))
print(target.shape)
print(preds.shape)

metric = JaccardIndex(task="multiclass", num_classes=3)
print(metric(preds, target))
