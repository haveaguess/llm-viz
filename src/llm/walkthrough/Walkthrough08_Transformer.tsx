import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { codeSnippet } from "../components/CodeSnippet";

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));

    let c0 = commentary(wt, null, 0)`

And that's a complete transformer block!

${codeSnippet(`class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x`, 'model.py — Block', 94)}

These form the bulk of any GPT model and are repeated a number of times, with the output of one
block feeding into the next, continuing the residual pathway.

${codeSnippet(`# In GPT.forward — loop through all transformer blocks:
for block in self.transformer.h:
    x = block(x)`, 'model.py — GPT.forward', 180)}

As is common in deep learning, it's hard to say exactly what each of these layers is doing, but we
have some general ideas: the earlier layers tend to focus on learning
lower-level features and patterns, while the later layers learn to recognize and understand
higher-level abstractions and relationships. In the context of natural language processing, the
lower layers might learn grammar, syntax, and simple word associations, while the higher layers
might capture more complex semantic relationships, discourse structures, and context-dependent meaning.

`;

}
