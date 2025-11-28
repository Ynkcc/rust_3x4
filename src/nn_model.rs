// code_files/src/nn_model.rs
use tch::{nn ,Tensor};

const BOARD_CHANNELS: i64 = 8;
const STATE_STACK: i64 = 1; // 禁用状态堆叠 (game_env.rs STATE_STACK_SIZE = 1)
const TOTAL_CHANNELS: i64 = BOARD_CHANNELS * STATE_STACK; // 8
const BOARD_H: i64 = 3;
const BOARD_W: i64 = 4;
const SCALAR_FEATURES: i64 = 56 * STATE_STACK; // 56
const ACTION_SIZE: i64 = 46;

pub struct BanqiNet {
    // Convolutional layers with residual blocks
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    conv3: nn::Conv2D,
    bn3: nn::BatchNorm,
    conv4: nn::Conv2D,
    bn4: nn::BatchNorm,
    conv5: nn::Conv2D,
    bn5: nn::BatchNorm,
    
    // Residual connections
    res_conv1: nn::Conv2D,  // for residual connection 1->3
    res_conv2: nn::Conv2D,  // for residual connection 3->5
    
    // Fully connected layers (combining conv output + scalars)
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    
    // Heads
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

impl BanqiNet {
    pub fn new(vs: &nn::Path) -> Self {
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        
        // Input: [Batch, 8, 3, 4] (禁用状态堆叠后)
        // Conv1: -> [Batch, 64, 3, 4]
        let conv1 = nn::conv2d(vs / "conv1", TOTAL_CHANNELS, 64, 3, conv_cfg);
        let bn1 = nn::batch_norm2d(vs / "bn1", 64, Default::default());
        
        // Conv2: -> [Batch, 128, 3, 4]
        let conv2 = nn::conv2d(vs / "conv2", 64, 128, 3, conv_cfg);
        let bn2 = nn::batch_norm2d(vs / "bn2", 128, Default::default());
        
        // Conv3: -> [Batch, 128, 3, 4]
        let conv3 = nn::conv2d(vs / "conv3", 128, 128, 3, conv_cfg);
        let bn3 = nn::batch_norm2d(vs / "bn3", 128, Default::default());
        
        // Conv4: -> [Batch, 256, 3, 4]
        let conv4 = nn::conv2d(vs / "conv4", 128, 256, 3, conv_cfg);
        let bn4 = nn::batch_norm2d(vs / "bn4", 256, Default::default());
        
        // Conv5: -> [Batch, 256, 3, 4]
        let conv5 = nn::conv2d(vs / "conv5", 256, 256, 3, conv_cfg);
        let bn5 = nn::batch_norm2d(vs / "bn5", 256, Default::default());
        
        // Residual connection conv layers (1x1 to match channels)
        let res_cfg = nn::ConvConfig { padding: 0, ..Default::default() };
        let res_conv1 = nn::conv2d(vs / "res_conv1", 64, 128, 1, res_cfg);  // conv1 -> conv3
        let res_conv2 = nn::conv2d(vs / "res_conv2", 128, 256, 1, res_cfg); // conv3 -> conv5
        
        let flat_size = 256 * BOARD_H * BOARD_W; // 256 * 3 * 4 = 3072
        let total_fc_input = flat_size + SCALAR_FEATURES; // 3072 + 56 = 3128
        
        let fc1 = nn::linear(vs / "fc1", total_fc_input, 1024, Default::default());
        let fc2 = nn::linear(vs / "fc2", 1024, 512, Default::default());
        let fc3 = nn::linear(vs / "fc3", 512, 256, Default::default());
        
        let policy_head = nn::linear(vs / "policy", 256, ACTION_SIZE, Default::default());
        let value_head = nn::linear(vs / "value", 256, 1, Default::default());
        
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            conv4,
            bn4,
            conv5,
            bn5,
            res_conv1,
            res_conv2,
            fc1,
            fc2,
            fc3,
            policy_head,
            value_head,
        }
    }
    
    pub fn forward_t(&self, board: &Tensor, scalars: &Tensor, train: bool) -> (Tensor, Tensor) {
        // Board path with residual connections
        // Block 1: conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu
        let x1 = board
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu();
        
        let x2 = x1
            .apply(&self.conv2)
            .apply_t(&self.bn2, train)
            .relu();
        
        // Residual block 1: conv3 with skip connection from x1
        let res1 = x1.apply(&self.res_conv1);
        let x3 = x2
            .apply(&self.conv3)
            .apply_t(&self.bn3, train);
        let x3 = (x3 + res1).relu();
        
        // Block 2: conv4 -> bn4 -> relu
        let x4 = x3
            .apply(&self.conv4)
            .apply_t(&self.bn4, train)
            .relu();
        
        // Residual block 2: conv5 with skip connection from x3
        let res2 = x3.apply(&self.res_conv2);
        let x5 = x4
            .apply(&self.conv5)
            .apply_t(&self.bn5, train);
        let x5 = (x5 + res2).relu();
            
        let x = x5.flatten(1, -1); // [Batch, 3072]
        
        // Concatenate scalars
        // Ensure scalars is [Batch, Features]
        let combined = Tensor::cat(&[&x, scalars], 1);
        
        // Deeper fully connected layers
        let shared = combined
            .apply(&self.fc1).relu()
            .apply(&self.fc2).relu()
            .apply(&self.fc3).relu();
            
        // Policy head: Logits (softmax applied later usually, or CrossEntropyLoss)
        // But MCTS expects probabilities. So we apply Softmax here for inference?
        // Usually output raw logits for training, softmax for MCTS. 
        // We will output logits here, handle conversion outside.
        let policy_logits = shared.apply(&self.policy_head);
        
        // Value head: Tanh activation for [-1, 1]
        let value = shared.apply(&self.value_head).tanh();
        
        (policy_logits, value)
    }

    pub fn forward(&self, board: &Tensor, scalars: &Tensor) -> (Tensor, Tensor) {
        self.forward_t(board, scalars, true)
    }
}