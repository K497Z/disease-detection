import torch
import torch.nn.functional as F

# This code implements the calculation of the SimCLR loss function. SimCLR is an unsupervised learning method used to learn feature representations of images. Its core idea is to optimize the model by contrasting different augmented views of the same image (positive pairs) and views of different images (negative pairs), making positive pairs closer in feature space and negative pairs further apart.
def compute_simclr_loss(logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):
    sim_aa = logits_a @ logits_a_gathered.t() / temperature # Dividing by temperature is mainly to adjust the distribution of similarity scores
    # Here dot product is used to compute similarity, similar to cosine similarity, representing the distance between images
    sim_ab = logits_a @ logits_b_gathered.t() / temperature
    sim_ba = logits_b @ logits_a_gathered.t() / temperature
    sim_bb = logits_b @ logits_b_gathered.t() / temperature
    # Used to compare similarity between different samples within the first group of augmented views. In subsequent loss calculation, masking is needed to exclude self-similarity scores to avoid the model wrongly learning the sample itself as a positive sample.
    # Positive pairs: Similarity scores of two augmented views of the same image (e.g., corresponding views in logits_a and logits_b for sample 1) are on the diagonal of sim_ab (e.g., sim_ab_11). Negative pairs: Similarity scores of all other samples (including intra-group and inter-group).
    masks = torch.where(F.one_hot(labels, logits_a_gathered.size(0)) == 0, 0, float('-inf'))# When the element in the one-hot encoding matrix is 0 (i.e., condition is True), the value at the corresponding position in the mask matrix is 0; when the element is 1 (i.e., condition is False), the value is negative infinity float('-inf'). In subsequent calculations, adding this mask matrix to the similarity score matrix ensures that positions corresponding to negative infinity will have probabilities approaching 0 during operations like softmax, thereby achieving the purpose of masking the sample's self-similarity score.
    # In contrastive learning, the model needs to distinguish between positive pairs (different augmented views of the same image) and negative pairs (views of different images)
    sim_aa += masks
    sim_bb += masks
    sim_a = torch.cat([sim_ab, sim_aa], 1)# This is also done to integrate similarity scores of positive pairs (sim_ba contains scores of different augmented views of the same image) and negative pairs (sim_bb contains similarity scores between the second group of augmented views of different images) for subsequent loss calculation, allowing the model to better distinguish between positive and negative pairs.
    sim_b = torch.cat([sim_ba, sim_bb], 1)
    loss_a = F.cross_entropy(sim_a, labels)# Labels marking positive pairs. In SimCLR, positive pairs refer to different augmented views of the same image, and labels are used to tell the model which are positive pairs.
    # Cross-Entropy Loss is commonly used in classification tasks to measure the difference between predicted probability distribution and true label distribution. In contrastive learning, it is cleverly used to force the model to maximize the similarity score of positive pairs and minimize the score of negative pairs.
    # Marking the position of positive pairs: For each sample i, the similarity score of its positive pair is located in the i-th column in sim_ab (assuming logits_b_gathered contains samples from all processes
    loss_b = F.cross_entropy(sim_b, labels)
    return (loss_a + loss_b) * 0.5
