import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

################################################################################
############################ TRANSFORMER CLASSES ###############################
################################################################################


class SelfAttention(nn.Module):

    def __init__(self, k, heads: int = 8):
        super().__init__()
        self.k, self.heads = k, heads

        # Create queries, keys and values. Use multihead attention.
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        self.unifyheads = nn.Linear(k * heads, k)

    def forward(self, x):
        b, t, k = x.size()
        heads = self.heads

        # Size from each linear layer is (b, t, k*heads)
        queries = self.toqueries(x).view(b, t, heads, k)
        keys = self.tokeys(x).view(b, t, heads, k)
        values = self.tovalues(x).view(b, t, heads, k)

        # Want to collapse heads and batch, need to keep time dimension intact to compute correct weights
        queries = queries.transpose(1, 2).contiguous().view(b * heads, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * heads, t, k)
        values = values.transpose(1, 2).contiguous().view(b * heads, t, k)

        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights,

        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights

        out = torch.bmm(dot, values).view(b, heads, t, k)

        # For each time point, have h k-dimensional outputs for each head!
        out = out.transpose(1, 2).contiguous().view(b, t, heads * k)

        # Output shape is (b, t, k)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # 'A SINGLE MLP APPLIED TO EACH VECTOR'
        self.mlp = nn.Sequential(nn.Linear(k, 4 * k),
                                 nn.ReLU(),
                                 nn.Linear(4 * k, k))

    def forward(self, x):
        # Part 1; self attention and normalize along feature direction (each sample own mean and var)
        out = self.attention(x)
        x = self.norm1(out + x)

        # Part 2; MLP and normalize along feature direction (each sample own mean and var)
        out = self.mlp(x)
        x = self.norm2(out + x)

        return x


class Transformer(nn.Module):
    def __init__(self, k, depth, heads, mlp_dim):
        super().__init__()

        # Transformer blocks
        blocks = []
        for i in range(depth):
            blocks.append(TransformerBlock(k, heads=heads))
        self.ff = nn.Sequential(*blocks)

        # Apply after average pooling over T dimension
        # self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        # Implement forward pass
        x = self.ff(x)
        # x = self.toprobs(x.mean(dim=1))
        # return F.log_softmax(x, dim=1)
        return x

class rearrange_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, p):
        return rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)


class VisionTransformer(nn.Module):
    """
    1. transform images into patches using einsum [b, c, 28, 28] -> [b, 4x4, 7x7xc]
    2. transform each 7x7xc vector into its embedding [b, 4x4, 7x7xc] -> [b, 4x4, dim]
    3. add a classification token to embedding [b, 4x4, dim] -> [b, 4x4 + 1, dim]
        use same token for each batch example (expand)
    4. add position embedding to embedded images [b, 4x4 + 1, dim] + [b, 4x4 + 1, dim]
    5. input into transformer [b, 4x4 + 1, dim] -> [b, 4x4 + 1, dim]
    6. extract token[b, 4x4 + 1, dim] -> [b, dim]
    7. input into head [b, dim] -> [b, num_classes]
    """

    def __init__(self, *, image_size, patch_size, num_classes, k, depth, heads, mlp_dim, channels=3):
        super().__init__()

        # image_size = 28, patch_size = 7,
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        # have 4x4 patches of 7x7xC images
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.rearrange_layer = rearrange_layer()
        self.patch_to_embedding = nn.Linear(patch_dim, k)
        self.cls_token = nn.Parameter(torch.randn(1, 1, k))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, k))
        self.transformer = Transformer(k, depth, heads, mlp_dim)
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(k, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = self.rearrange_layer(img, p)
        #print(x.shape)
        x = self.patch_to_embedding(x)
        #print(x.shape)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        #print(x.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        #print(x.shape)
        x += self.pos_embedding
        #print(x.shape)
        x = self.transformer(x)
        #print(x.shape)

        x = self.to_cls_token(x[:, 0])
        #print(x.shape)
        return self.mlp_head(x)

################################################################################
################################################################################

