import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { codeSnippet } from "../components/CodeSnippet";

export function walkthrough09_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Output) {
        return;
    }

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

    let c0 = commentary(wt, null, 0)`

Finally, we come to the end of the model. The output of the final transformer block is passed through
a final layer normalization (\`self.transformer.ln_f\`), and then we use a linear transformation (\`self.lm_head\`), this time without a bias.

${codeSnippet(`x = self.transformer.ln_f(x)          # (1, 11, 48)
logits = self.lm_head(x)              # (1, 11, 3) — one score per vocab token`, 'model.py — GPT.forward', 182)}

This final transformation takes each of our column vectors from length C to length nvocab. Hence,
it's effectively producing a score for each word in the vocabulary for each of our columns. These
scores have a special name: logits (\`logits\`).

The name "logits" comes from "log-odds," i.e., the logarithm of the odds of each token. "Log" is
used because the softmax we apply next does an exponentiation to convert to "odds" or probabilities.

To convert these scores into nice probabilities, we pass them through a softmax operation. Now, for
each column, we have a probability the model assigns to each word in the vocabulary.

In this particular model, it has effectively learned all the answers to the question of how to sort
three letters, so the probabilities are heavily weighted toward the correct answer.

When we're stepping the model through time, we use the last column's probabilities to determine the
next token to add to the sequence. For example, if we've supplied six tokens into the model, we'll
use the output probabilities of the 6th column.

This column's output is a series of probabilities, and we actually have to pick one of them to use
as the next in the sequence. We do this by "sampling from the distribution." That is, we randomly
choose a token, weighted by its probability. For example, a token with a probability of 0.9 will be
chosen 90% of the time.

${codeSnippet(`# GPT.generate — sampling the next token:
logits = logits[:, -1, :] / temperature    # (1, 3)
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
probs = F.softmax(logits, dim=-1)          # (1, 3) probabilities
idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
idx = torch.cat((idx, idx_next), dim=1)    # append to sequence`, 'model.py — GPT.generate', 318)}

There are other options here, however, such as always choosing the token with the highest probability.

We can also control the "smoothness" of the distribution by using a temperature parameter. A higher
temperature will make the distribution more uniform, and a lower temperature will make it more
concentrated on the highest probability tokens.

We do this by dividing the logits (the output of the linear transformation) by the temperature before
applying the softmax. Since the exponentiation in the softmax has a large effect on larger numbers,
making them all closer together will reduce this effect.

During training, we compute the loss using cross-entropy:

${codeSnippet(`# In GPT.forward — computing the training loss:
if targets is not None:
    logits = self.lm_head(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1), ignore_index=-1)`, 'model.py — GPT.forward', 184)}
`;

}
