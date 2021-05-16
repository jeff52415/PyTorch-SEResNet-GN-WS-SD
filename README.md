# SEResNet - BN, GN, WS, SD
--------------
</div>

## Available Models
- [x] senet154
- [x] se_resnet50
- [x] se_resnet101
- [x] se_resnet152
- [x] se_resnext50_32x4d
- [x] se_resnext101_32x4d



--------------
</div>


## Installation

```bash
pip install -e .
```




--------------
</div>

## Usage

```python
from senet_pack.models import se_resnext50_32x4d
import torch

dummy_ = torch.randn(32, 3, 64, 64)
model = se_resnext50_32x4d(num_classes=10, pretrained=None, group_normalization=False, weight_standardization=False, stochastic_depth=False)
output = model(dummy_)
```
