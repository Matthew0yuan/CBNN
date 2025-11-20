import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Config

IMG_CHANNELS = 1      #MNIST: 1
IMG_SIZE = 28        #MNIST: 28
NUM_CLASSES = 10

D_MODEL = 64         # Feature dimension / Cell output dimension
NUM_CNN_CELLS = 4
NUM_TRANS_CELLS = 4
NUM_MLP_CELLS = 4
TOP_K = 4            # Max number of active Cells per image (cell-level)

HEBB_LR = 1e-3       # Local Hebbian learning rate for Cells
CLASS_LR = 5e-3      # Hebbian learning rate for classifier head
ROUTER_LR = 5e-3     # Learning rate for router keys

BATCH_SIZE = 128
NUM_EPOCHS = 50       # Number of demo training epochs

#Router anti-collapse parameters (slightly gentle)
USAGE_MU = 0.015      # EMA update coefficient for usage
USAGE_PENALTY = 0.02  # Usage penalty strength (applied to both groups and cells)

#Router temperature & exploration parameters
TEMP_START = 1.0
TEMP_END = 0.3
TEMP_DECAY_STEPS = 20000

EPS_START = 0.2       #Starting epsilon for exploration
EPS_END = 0.05
EPS_DECAY_STEPS = 10000

#Dynamic rewiring parameters (weakened)
REWIRE_START_EPOCH = 8     #Start rewiring from this epoch
REWIRE_PER_EPOCH = 1       #How many cells to rewire per epoch
ELITE_FRAC = 0.25          #Top 25% cells by usage are elite and never rewired


#Generic Hebbian linear layer (no learning in forward)

class LocalHebbianLinear(nn.Module):
    def __init__(self, in_dim, out_dim, hebb_lr=HEBB_LR):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(in_dim, out_dim) * 0.02,
            requires_grad=False
        )
        self.hebb_lr = hebb_lr
        self.ln = nn.LayerNorm(out_dim)

        #Store last batch input/output for updates
        self.last_x = None
        self.last_y = None

    def forward(self, x):
        """
        x: [B, in_dim]
        return: [B, out_dim]
        """
        y = x @ self.W   #[B, out_dim]
        y = self.ln(y)

        #Record for later updates
        self.last_x = x.detach()
        self.last_y = y.detach()
        return y

    @torch.no_grad()
    def hebbian_unsup_update(self, scale=1.0):
        """
        Pure unsupervised Hebbian: ΔW ∝ x^T y
        By default can be unused or use very small scale.
        """
        if self.last_x is None or self.last_y is None:
            return
        x = self.last_x
        y = self.last_y
        B = x.size(0)
        dW = (x.t() @ y) / (B + 1e-6)
        self.W += self.hebb_lr * scale * dW

        # Simple normalization to avoid explosion
        norm = self.W.norm()
        if norm > 1.0:
            self.W /= norm

    @torch.no_grad()
    def reward_update(self, reward, scale=1.0):
        """
        reward: [B] or [B,1]
        Three-factor rule: ΔW ∝ x^T (reward * y)
        """
        if self.last_x is None or self.last_y is None:
            return

        x = self.last_x        # [B_last, in_dim]
        y = self.last_y        # [B_last, out_dim]
        B_last = x.size(0)

        B_reward = reward.size(0)
        B = min(B_last, B_reward)
        if B == 0:
            return

        x = x[:B]
        y = y[:B]

        r = reward[:B].view(B, 1)  # [B,1]
        modulated_post = r * y     # [B, out_dim]

        dW = (x.t() @ modulated_post) / (B + 1e-6)
        self.W += self.hebb_lr * scale * dW

        # Normalize
        norm = self.W.norm()
        if norm > 1.0:
            self.W /= norm

    @torch.no_grad()
    def supervised_update(self, labels, scale=1.0):
        """
        Only used when this layer is used as a classification head.
        """
        if self.last_x is None or self.last_y is None:
            return

        x = self.last_x        # [B,in_dim]
        y = self.last_y        # [B,out_dim]
        B_last = x.size(0)
        B_labels = labels.size(0)
        B = min(B_last, B_labels)
        if B == 0:
            return

        x = x[:B]
        y = y[:B]
        labels = labels[:B]

        out_dim = y.size(1)
        t = torch.zeros(B, out_dim, device=x.device)
        t[torch.arange(B), labels] = 1.0  # one-hot

        error = t - y   # [B,out_dim]
        dW = (x.t() @ error) / (B + 1e-6)
        self.W += self.hebb_lr * scale * dW

        # Normalize
        norm = self.W.norm()
        if norm > 1.0:
            self.W /= norm


# ===========================
# CNN Cell
# ===========================
class CNNCell(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        # Conv is not trained via BP
        for p in self.conv.parameters():
            p.requires_grad = False

        self.readout = LocalHebbianLinear(16, d_model)

    def forward(self, x):
        feat = self.conv(x)          # [B,16,H,W]
        feat = F.relu(feat)
        feat = feat.mean(dim=[2, 3]) # GAP -> [B,16]
        h = self.readout(feat)
        return h


# ===========================
# Transformer-like Cell
# ===========================
class TransformerCell(nn.Module):
    def __init__(self, in_channels, img_size, d_model, patch_size=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model

        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding (fixed random)
        self.patch_embed = nn.Linear(patch_dim, d_model)
        for p in self.patch_embed.parameters():
            p.requires_grad = False

        # Self-attention weights (fixed)
        self.W_q = nn.Parameter(
            torch.randn(d_model, d_model) * 0.02,
            requires_grad=False
        )
        self.W_k = nn.Parameter(
            torch.randn(d_model, d_model) * 0.02,
            requires_grad=False
        )
        self.W_v = nn.Parameter(
            torch.randn(d_model, d_model) * 0.02,
            requires_grad=False
        )

        self.readout = LocalHebbianLinear(d_model, d_model)

    def _img_to_patches(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patches = F.unfold(x, kernel_size=p, stride=p)  # [B, C*p*p, N]
        patches = patches.transpose(1, 2)               # [B, N, patch_dim]
        return patches

    def forward(self, x):
        B = x.size(0)
        patches = self._img_to_patches(x)         # [B,N,patch_dim]
        tokens = self.patch_embed(patches)        # [B,N,d_model]

        q = tokens @ self.W_q
        k = tokens @ self.W_k
        v = tokens @ self.W_v

        attn_scores = q @ k.transpose(1, 2) / math.sqrt(self.d_model)
        attn = F.softmax(attn_scores, dim=-1)
        out = attn @ v                           # [B,N,d_model]

        pooled = out.mean(dim=1)                 # [B,d_model]
        h = self.readout(pooled)
        return h


# ===========================
# Simple MLP Cell
# ===========================
class MLPCell(nn.Module):
    def __init__(self, in_channels, img_size, d_model):
        super().__init__()
        in_dim = in_channels * img_size * img_size
        self.readout = LocalHebbianLinear(in_dim, d_model)

    def forward(self, x):
        B = x.size(0)
        flat = x.view(B, -1)
        h = self.readout(flat)
        return h


# ===========================
# Hierarchical Router (Group + Cell), no BP, only reward_update
# ===========================
class HierarchicalRouter(nn.Module):
    """
    Hierarchical routing:
    - group-level: compute probability pg for each group (soft gating)
    - cell-level: pc is masked by pg[:, group_ids], then do cell top-k
    - both levels use three-factor RL updates (reward * gate * embed)
    """
    def __init__(self,
                 in_channels,
                 img_size,
                 d_model,
                 num_cells,
                 group_ids,                # LongTensor[num_cells], group id for each cell
                 top_k_cells=TOP_K,
                 lr=ROUTER_LR):
        super().__init__()
        self.num_cells = num_cells
        self.lr = lr

        # group info
        if not torch.is_tensor(group_ids):
            group_ids = torch.tensor(group_ids, dtype=torch.long)
        num_groups = int(group_ids.max().item()) + 1
        self.num_groups = num_groups
        self.top_k_cells = top_k_cells

        # register group_ids as buffer to follow .to(device)
        self.register_buffer("group_ids", group_ids)

        # router embedding: Conv -> GAP -> Linear (fixed random)
        self.embed_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        for p in self.embed_conv.parameters():
            p.requires_grad = False

        self.embed_proj = nn.Linear(8, d_model)
        for p in self.embed_proj.parameters():
            p.requires_grad = False

        # group keys & cell keys (all updated by plasticity rules)
        self.group_keys = nn.Parameter(
            torch.randn(num_groups, d_model) * 0.02,
            requires_grad=False
        )
        self.cell_keys = nn.Parameter(
            torch.randn(num_cells, d_model) * 0.02,
            requires_grad=False
        )

        # usage EMA
        group_usage_init = torch.ones(num_groups) / num_groups
        cell_usage_init = torch.ones(num_cells) / num_cells
        self.register_buffer("group_usage_ema", group_usage_init.clone())
        self.register_buffer("cell_usage_ema", cell_usage_init.clone())
        #store last forward embed & gates
        self.last_embed = None           # [B, d_model]
        self.last_group_gates = None     # [B, num_groups] (soft pg)
        self.last_cell_gates = None      # [B, num_cells]
        #training steps & temperature / exploration
        self.step = 0
        self.temperature = TEMP_START
        self.eps = EPS_START
        # Router reward baseline (loss_old)
        self.loss_baseline = None

    def _update_sched(self):
        # Linear decay of temperature & epsilon
        t_ratio = min(1.0, self.step / TEMP_DECAY_STEPS)
        self.temperature = TEMP_START + (TEMP_END - TEMP_START) * t_ratio

        e_ratio = min(1.0, self.step / EPS_DECAY_STEPS)
        self.eps = EPS_START + (EPS_END - EPS_START) * e_ratio

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: gates [B, num_cells]
        """
        B = x.size(0)
        self.step += 1
        self._update_sched()

        #Extract embedding
        feat = self.embed_conv(x)          # [B,8,H,W]
        feat = F.relu(feat)
        feat = feat.mean(dim=[2, 3])       # GAP -> [B,8]
        embed = self.embed_proj(feat)      # [B,d_model]
        embed_norm = F.normalize(embed, dim=-1)          # [B,d_model]

        #Group-level routing soft pg
        group_keys_norm = F.normalize(self.group_keys, dim=-1)   # [G,d_model]
        sim_g = embed_norm @ group_keys_norm.t()                 # [B,G]

        #group usage penalty
        group_usage_penalty = USAGE_PENALTY * (self.group_usage_ema - self.group_usage_ema.mean())
        sim_g = sim_g - group_usage_penalty.unsqueeze(0)         # [B,G]

        logits_g = sim_g / self.temperature
        base_pg = F.softmax(logits_g, dim=-1)                    # [B,G]

        if self.training:
            explore_mask_g = (torch.rand(B, 1, device=x.device) < self.eps).float()
            uniform_g = torch.full_like(base_pg, 1.0 / self.num_groups)
            pg = explore_mask_g * uniform_g + (1.0 - explore_mask_g) * base_pg
        else:
            pg = base_pg
        group_gates = pg    # soft group gate

        #Cell-level routing first probability, modulate by group soft gate
        cell_keys_norm = F.normalize(self.cell_keys, dim=-1)          # [C,d_model]
        sim_c = embed_norm @ cell_keys_norm.t()                       # [B,C]

        cell_usage_penalty = USAGE_PENALTY * (self.cell_usage_ema - self.cell_usage_ema.mean())
        sim_c = sim_c - cell_usage_penalty.unsqueeze(0)               # [B,C]

        logits_c = sim_c / self.temperature
        base_pc = F.softmax(logits_c, dim=-1)                         # [B,C]

        if self.training:
            explore_mask_c = (torch.rand(B, 1, device=x.device) < self.eps).float()
            uniform_c = torch.full_like(base_pc, 1.0 / self.num_cells)
            pc = explore_mask_c * uniform_c + (1.0 - explore_mask_c) * base_pc
        else:
            pc = base_pc

        #soft hierarchy: cell probs multiplied by group prob
        soft_group_per_cell = group_gates[:, self.group_ids]          # [B,C]
        pc_masked = pc * soft_group_per_cell
        pc_masked = pc_masked / (pc_masked.sum(dim=-1, keepdim=True) + 1e-6)

        tc_vals, tc_idx = torch.topk(pc_masked, self.top_k_cells, dim=-1)
        hard_c = torch.zeros_like(pc_masked)
        hard_c.scatter_(1, tc_idx, 1.0)
        hard_c = hard_c / (hard_c.sum(dim=-1, keepdim=True) + 1e-6)   # [B,C]

        gates = hard_c

        #Update usage EMA
        with torch.no_grad():
            cell_usage_batch = (gates > 0).float().mean(dim=0)        # [C]
            group_usage_batch = group_gates.mean(dim=0)               # [G] soft usage
            self.cell_usage_ema = (1 - USAGE_MU) * self.cell_usage_ema + USAGE_MU * cell_usage_batch
            self.group_usage_ema = (1 - USAGE_MU) * self.group_usage_ema + USAGE_MU * group_usage_batch

        #Store embed gates
        self.last_embed = embed_norm.detach()
        self.last_group_gates = group_gates.detach()  # soft
        self.last_cell_gates = gates.detach()

        return gates

    @torch.no_grad()
    def reward_update(self, reward):
        """
        reward: [B], e.g. (loss_old - loss_new), already detached
        For groups and cells: Δkey ∝ E_b[ (reward_b * gate_b) * embed_b ]
        """
        if self.last_embed is None or self.last_cell_gates is None or self.last_group_gates is None:
            return

        embed = self.last_embed           # [B_embed,d_model]
        gates_c = self.last_cell_gates    # [B_embed,num_cells]
        gates_g = self.last_group_gates   # [B_embed,num_groups] (soft)

        B_embed = embed.size(0)
        B_reward = reward.size(0)
        B = min(B_embed, B_reward)
        if B == 0:
            return

        embed = embed[:B]                 # [B,d_model]
        gates_c = gates_c[:B]             # [B,C]
        gates_g = gates_g[:B]             # [B,G]
        r = reward[:B].view(B, 1)         # [B,1]

        #Cell-level key update
        eff_c = r * gates_c               # [B,C]
        eff_c_centered = eff_c - eff_c.mean(dim=0, keepdim=True)
        dK_c = eff_c_centered.t() @ embed  # [C,d_model]
        dK_c = dK_c / (B + 1e-6)
        self.cell_keys += self.lr * dK_c

        cell_norm = self.cell_keys.norm(dim=1, keepdim=True) + 1e-6
        self.cell_keys /= cell_norm

        #Group-level key update
        eff_g = r * gates_g               # [B,G]
        eff_g_centered = eff_g - eff_g.mean(dim=0, keepdim=True)
        dK_g = eff_g_centered.t() @ embed  # [G,d_model]
        dK_g = dK_g / (B + 1e-6)
        self.group_keys += self.lr * dK_g

        group_norm = self.group_keys.norm(dim=1, keepdim=True) + 1e-6
        self.group_keys /= group_norm



#Linear classifier head (Softmax Hebbian, no BP)

class SoftmaxHebbianClassifier(nn.Module):
    def __init__(self, d_model, num_classes, lr=CLASS_LR):
        super().__init__()
        # Bare W, no LayerNorm
        self.W = nn.Parameter(
            torch.randn(d_model, num_classes) * 0.01,
            requires_grad=False
        )
        self.lr = lr
        self.last_h = None        # [B, d_model]
        self.last_logits = None   # [B, num_classes]

    def forward(self, h):
        """
        h: [B, d_model]
        return: logits [B, num_classes]
        """
        logits = h @ self.W       # [B, num_classes]
        # Record for local update later
        self.last_h = h.detach()
        self.last_logits = logits.detach()
        return logits

    @torch.no_grad()
    def supervised_update(self, labels, scale=1.0):
        """
        Local gradient of CE:
        where p = softmax(logits), t is one-hot(label)
        """
        if self.last_h is None or self.last_logits is None:
            return

        h = self.last_h          # [B, d_model]
        logits = self.last_logits  # [B, num_classes]
        B = h.size(0)
        if B == 0:
            return

        # softmax probabilities
        p = F.softmax(logits, dim=-1)  # [B, num_classes]

        # one-hot labels
        t = torch.zeros_like(p)
        labels = labels[:B]
        t[torch.arange(B), labels] = 1.0

        #CE gradient: t - p
        err = t - p                # [B, num_classes]

        #local update
        dW = h.t() @ err / (B + 1e-6)  # [d_model, num_classes]
        self.W += self.lr * scale * dW

        #Avoid explosion
        norm = self.W.norm()
        if norm > 5.0:
            self.W *= 5.0 / norm



#Full CBNN image model (with hierarchical Router)

class CBNNImageModel(nn.Module):
    def __init__(self,
                 img_channels=IMG_CHANNELS,
                 img_size=IMG_SIZE,
                 d_model=D_MODEL,
                 num_classes=NUM_CLASSES,
                 num_cnn=NUM_CNN_CELLS,
                 num_trans=NUM_TRANS_CELLS,
                 num_mlp=NUM_MLP_CELLS,
                 top_k=TOP_K):
        super().__init__()

        cells = []
        cell_groups = []  #group id per cell: 0/1/2 for CNN/Trans/MLP (only for existing types)

        group_idx = 0
        # CNN group
        if num_cnn > 0:
            for _ in range(num_cnn):
                cells.append(CNNCell(img_channels, d_model))
                cell_groups.append(group_idx)
            group_idx += 1

        # Transformer group
        if num_trans > 0:
            for _ in range(num_trans):
                cells.append(TransformerCell(img_channels, img_size, d_model))
                cell_groups.append(group_idx)
            group_idx += 1

        # MLP group
        if num_mlp > 0:
            for _ in range(num_mlp):
                cells.append(MLPCell(img_channels, img_size, d_model))
                cell_groups.append(group_idx)
            group_idx += 1

        self.cells = nn.ModuleList(cells)
        self.num_cells = len(cells)

        group_ids = torch.tensor(cell_groups, dtype=torch.long)

        self.router = HierarchicalRouter(
            img_channels, img_size, d_model,
            self.num_cells,
            group_ids=group_ids,
            top_k_cells=top_k,
            lr=ROUTER_LR,
        )
        # SoftmaxHebbian classifier head
        self.classifier = SoftmaxHebbianClassifier(d_model, num_classes, lr=CLASS_LR)

        # Frozen slow backbone: used as a stable anchor (still random)
        self.slow_conv = nn.Conv2d(img_channels, 8, kernel_size=3, padding=1)
        self.slow_proj = nn.Linear(8, d_model)
        for p in self.slow_conv.parameters():
            p.requires_grad = False
        for p in self.slow_proj.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: logits, h_total, gates
        """
        B = x.size(0)

        gates = self.router(x)      # [B,num_cells]

        cell_outputs = []
        for i, cell in enumerate(self.cells):
            gate_i = gates[:, i].unsqueeze(-1)  # [B,1]
            if gate_i.max().item() == 0.0:
                cell_outputs.append(torch.zeros(B, D_MODEL, device=x.device))
                continue
            hi = cell(x)            # [B,d_model]
            cell_outputs.append(hi * gate_i)

        h_mix = torch.stack(cell_outputs, dim=0).sum(dim=0)   # [B,d_model]

        # Frozen slow backbone as anchor
        slow_feat = self.slow_conv(x)                         # [B,8,H,W]
        slow_feat = F.relu(slow_feat)
        slow_feat = slow_feat.mean(dim=[2, 3])                # [B,8]
        slow_h = self.slow_proj(slow_feat)                    # [B,d_model]

        h_total = h_mix + slow_h                              # [B,d_model]

        logits = self.classifier(h_total)                     # classifier head (no BP)
        return logits, h_total, gates


# ===========================
# Data
# ===========================
def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    return train_loader, test_loader



#Evaluation (no updates)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits, h_total, gates = model(imgs)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc



#Visualization: which Cells each class uses

def visualize_cell_usage_per_class(model, loader):
    model.eval()
    num_cells = model.num_cells
    num_classes = NUM_CLASSES

    usage_matrix = torch.zeros(num_classes, num_cells)
    count_per_class = torch.zeros(num_classes)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            _, _, gates = model(imgs)        # [B,num_cells]

            B = imgs.size(0)
            for c in range(num_classes):
                mask = (labels == c).float().unsqueeze(-1)  # [B,1]
                if mask.sum() == 0:
                    continue
                gates_c = (gates * mask).sum(dim=0) / (mask.sum() + 1e-6)  # [num_cells]
                usage_matrix[c] += gates_c.cpu()
                count_per_class[c] += 1.0

    for c in range(num_classes):
        if count_per_class[c] > 0:
            usage_matrix[c] /= count_per_class[c]

    plt.figure(figsize=(8, 6))
    plt.imshow(usage_matrix.numpy(), aspect='auto')
    plt.colorbar(label='Average gate value')
    plt.xlabel('Cell index')
    plt.ylabel('Digit class')
    plt.title('Cell usage per digit class (average gates)')
    plt.yticks(range(num_classes), [str(i) for i in range(num_classes)])
    plt.show()


#Dynamic rewiring: rewire least-used cells

@torch.no_grad()
def dynamic_rewire_least_used_cells(model,
                                    num_rewire: int = REWIRE_PER_EPOCH,
                                    elite_frac: float = ELITE_FRAC):
    """
    Dynamic rewiring based on router.cell_usage_ema:
    - Keep a fraction of top-usage "elite cells" unchanged
    - Among the remaining cells, select the lowest-usage ones
    - For these cells:
        * Reset readout.W (like a new expert)
        * Reset router.cell_keys[i] (new direction for router)
        * Reset cell_usage_ema[i]
    """
    router = model.router
    usage = router.cell_usage_ema.clone()           # [num_cells]
    num_cells = model.num_cells

    if num_cells <= 1 or num_rewire <= 0:
        return

    #Elite cells (highest usage), do not touch
    num_elite = max(1, int(num_cells * elite_frac))
    num_elite = min(num_elite, num_cells)
    elite_values, elite_idx = torch.topk(usage, num_elite, largest=True)

    #Remaining candidates, sorted by usage ascending
    mask = torch.ones(num_cells, dtype=torch.bool, device=usage.device)
    mask[elite_idx] = False
    candidate_idx = torch.arange(num_cells, device=usage.device)[mask]
    if candidate_idx.numel() == 0:
        return

    candidate_usage = usage[mask]
    _, order = torch.sort(candidate_usage)     #ascending: least used first

    num_rewire = min(num_rewire, candidate_idx.numel())
    chosen = candidate_idx[order[:num_rewire]]  #indices of cells to rewire

    #Rebirthfor these cells
    for idx in chosen:
        i = int(idx.item())
        cell = model.cells[i]

        #Reset cell.readout weights (if it has one)
        if hasattr(cell, "readout"):
            W = cell.readout.W
            in_dim, out_dim = W.shape
            W.copy_(torch.randn(in_dim, out_dim, device=W.device) * 0.02)
            cell.readout.last_x = None
            cell.readout.last_y = None

        #Reset router cell key
        k = router.cell_keys[i]
        k.copy_(torch.randn_like(k) * 0.02)

        #Reset usage_ema
        router.cell_usage_ema[i] = usage.mean()

    print(f"[Dynamic rewiring] Rewired cells: {[int(i.item()) for i in chosen]}")


# ===========================
# Training (pure Hebbian + Hierarchical Router RL + Dynamic rewiring, no BP)
# ===========================
def train_mnist():
    train_loader, test_loader = get_mnist_loaders()
    model = CBNNImageModel().to(DEVICE)

    usage_ema_history = []  # record cell_usage_ema at end of each epoch

    step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward
            logits, h_total, gates = model(imgs)

            # per-sample loss for reward & logging
            loss_vec = F.cross_entropy(logits, labels, reduction="none")
            loss = loss_vec.mean()

            # Router reward = (loss_old - loss_new)
            with torch.no_grad():
                if model.router.loss_baseline is None:
                    model.router.loss_baseline = loss.detach().item()
                baseline = model.router.loss_baseline

                adv = baseline - loss_vec.detach()  # [B], lower loss → higher adv (positive reward)
                model.router.loss_baseline = 0.9 * baseline + 0.1 * loss_vec.detach().mean().item()

                reward = adv

            #Classifier head: supervised Hebbian update
            model.classifier.supervised_update(labels, scale=1.0)

            #Use reward * gate to modulate Hebbian update of each Cell
            with torch.no_grad():
                B, num_cells = gates.shape
                for i, cell in enumerate(model.cells):
                    gate_i = gates[:, i]  # [B]
                    if gate_i.max().item() == 0.0:
                        continue
                    effective_reward = reward * gate_i  # [B]
                    cell.readout.reward_update(effective_reward, scale=1.0)
                    # -----If you want some extra unsupervised Hebbian,below with small scale:
                    # cell.readout.hebbian_unsup_update(scale=0.05)

            #Router plasticity update based on reward
            if epoch >= 1:  # let classifier & cells adapt in first epoch, then update router
                model.router.reward_update(reward)

            step += 1
            if step % 100 == 0:
                with torch.no_grad():
                    usage_batch = (gates.detach() > 0).float().mean(dim=0)
                    usage_str = " ".join([f"{u.item():.2f}" for u in usage_batch])
                print(f"[epoch {epoch+1} step {step}] loss={loss.item():.4f}")
                print("  cell usage (this batch):", usage_str)

        #run test
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Test accuracy: {acc*100:.2f}%")

        #Record current cell_usage_ema
        usage_ema_history.append(model.router.cell_usage_ema.cpu().clone())

        #Dynamic rewiring: from REWIRE_START_EPOCH, periodically rewire lowest-usage cells
        if (epoch + 1) >= REWIRE_START_EPOCH:
            dynamic_rewire_least_used_cells(
                model,
                num_rewire=REWIRE_PER_EPOCH,
                elite_frac=ELITE_FRAC,
            )

    print("Training finished.")

    #Plot evolution of cell_usage_ema
    usage_ema_history_tensor = torch.stack(usage_ema_history, dim=0)  # [E,num_cells]
    plt.figure(figsize=(8, 4))
    for i in range(model.num_cells):
        plt.plot(
            range(1, NUM_EPOCHS + 1),
            usage_ema_history_tensor[:, i].numpy(),
            label=f"Cell {i}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("cell_usage_ema")
    plt.title("Router cell_usage_ema per cell over epochs")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    #Visualize which cells each digit uses
    visualize_cell_usage_per_class(model, test_loader)

#by MatthewYuan 
if __name__ == "__main__":
    for i in range(3):
        train_mnist()
        print("test1:", i)
        

        

