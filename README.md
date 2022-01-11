# Day 1, Advent of Code 2021

This task is to just count the number of times consecutive pairs of numbers in a list have the latter number larger than the former.  This is obviously very easy to code a direct solution to.  However, learning the solution from data is not straightforward unless we build too much knowledge into our network to begin with.

We could generate training data of the form:
 * input: variable-length list of integers
 * output: integer

This has low inductive bias (we don't build much of anything into our neural network), but is likely difficult and unstable to train due to long-range dependencies (the input data is 2,000 numbers long).

Alternatively, we could exploit our knowledge of the problem to focus on individual pairs in our data.  This has high inductive bias.

Under any approach, I expect difficulties with the scale of the input data (numbers seem to range in the 1000s).  Obviously we can normalize our input data, but this is obviously building more knowledge into our neural network as we're exploiting the fact that the problem only depends on the relative values of the input data.

## Part 1

We split our solutions into various approaches, ordered roughly from strong to weak inductive biases in our model.  Even models with high inductive bias are interesting because they isolate specific aspects of learning problem (e.g. the pairwise approach will indicate how much the input data scale impacts the problem).
- __pairwise__: instead of treating the data as a list, we get the model to classify pairs of data points as higher or lower.
  - __pairwise_and_sum__: We can then sum that output directly to produce our final answer, which is obviously exploiting knowledge of the solution.
  - __pairwise_and_count__: Or we can feed the output of the pairwise model into a counting sequence model.

- __sequential__: this takes in a variable-length list of numbers and produces the final answer to the problem.  
  - __seq2int__: We can produce _just_ the final answer by prompting the network with a special character to indicate the sequence has finished.
  - __seq2seq__: Or we can give the model feedback at each timestep of the count so far.
