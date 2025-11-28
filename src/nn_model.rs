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
    // Convolutional layers
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    
    // Fully connected layers (combining conv output + scalars)
    fc1: nn::Linear,
    fc2: nn::Linear,
    
    // Heads
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

impl BanqiNet {
    pub fn new(vs: &nn::Path) -> Self {
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        
        // Input: [Batch, 8, 3, 4] (禁用状态堆叠后)
        // Conv1: -> [Batch, 32, 3, 4]
        let conv1 = nn::conv2d(vs / "conv1", TOTAL_CHANNELS, 32, 3, conv_cfg);
        // Conv2: -> [Batch, 64, 3, 4]
        let conv2 = nn::conv2d(vs / "conv2", 32, 64, 3, conv_cfg);
        // Conv3: -> [Batch, 64, 3, 4] (keeping size)
        let conv3 = nn::conv2d(vs / "conv3", 64, 64, 3, conv_cfg);
        
        let flat_size = 64 * BOARD_H * BOARD_W; // 64 * 3 * 4 = 768
        let total_fc_input = flat_size + SCALAR_FEATURES; // 768 + 56 = 824
        
        let fc1 = nn::linear(vs / "fc1", total_fc_input, 512, Default::default());
        let fc2 = nn::linear(vs / "fc2", 512, 256, Default::default());
        
        let policy_head = nn::linear(vs / "policy", 256, ACTION_SIZE, Default::default());
        let value_head = nn::linear(vs / "value", 256, 1, Default::default());
        
        Self {
            conv1,
            conv2,
            conv3,
            fc1,
            fc2,
            policy_head,
            value_head,
        }
    }
    
    pub fn forward(&self, board: &Tensor, scalars: &Tensor) -> (Tensor, Tensor) {
        // Board path
        let x = board
            .apply(&self.conv1).relu()
            .apply(&self.conv2).relu()
            .apply(&self.conv3).relu();
            
        let x = x.flatten(1, -1); // [Batch, 768]
        
        // Concatenate scalars
        // Ensure scalars is [Batch, Features]
        let combined = Tensor::cat(&[&x, scalars], 1);
        
        let shared = combined
            .apply(&self.fc1).relu()
            .apply(&self.fc2).relu();
            
        // Policy head: Logits (softmax applied later usually, or CrossEntropyLoss)
        // But MCTS expects probabilities. So we apply Softmax here for inference?
        // Usually output raw logits for training, softmax for MCTS. 
        // We will output logits here, handle conversion outside.
        let policy_logits = shared.apply(&self.policy_head);
        
        // Value head: Tanh activation for [-1, 1]
        let value = shared.apply(&self.value_head).tanh();
        
        (policy_logits, value)
    }
}