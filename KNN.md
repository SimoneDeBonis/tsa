---
layout: post
---

## K-Nearest-Neighboor

Given a region of space L(X) centered on X, of volume v, the probability is approximable as p(X)v and can be estimated as:
<p style="text-align: center;">$$
\hat{p}(X)v=\frac{k(X)}{N}^\text{N of instances in L(x)}_\text{N of instances} \hat{p}(X)= \frac{k(X)}{Nv}
$$</p>
Requires:
- A set of labeled records;
- A proximity metric to compute distances/similarities;
- k, the number o nearest neighbours (volumetric);
- A method for using class labels of K nearest neighbours to determine the class label of the new record (e.g. majority vote).


The majority vote rule can be improved considering the inverse of the square of the distance as weight for each vote.

*Voting KNN*

From $$P(X)=\frac{k(x)}{Nv}$$ we can build a bayes classifier with $$p(\omega_i) = \frac{N_i}{N}$$

<p style="text-align: center;">$$g(x) = \begin{cases} \omega_1 \quad if \quad \frac{N_1}{N}\frac{k(X|\omega_1)}{N_1v}> \frac{N_2}{N}\frac{k(X|\omega_2)}{N_2v}\\ 
\omega_2 \quad if \quad \frac{N_1}{N}\frac{k(X|\omega_1)}{N_1v}< \frac{N_2}{N}\frac{k(X|\omega_2)}{N_2v}  \end{cases}$$</p>

So: 
<p style="text-align: center;">$$g(x) = \begin{cases} \omega_1 \quad if \quad k(X|\omega_1)> k(X|\omega_2)\\ 
    \omega_2 \quad if \quad k(X|\omega_1)< k(X|\omega_2)  \end{cases}$$</p>

*Volumetric KNN*

Fixing k, the volume of the region centered in X depends on X, $P(x) = \frac{k}{Nv(x)}$

<p style="text-align: center;">$$g(x) = \begin{cases} \omega_1 \quad if \quad \frac{N_1}{N}\frac{k}{N_1v_1(X)}> \frac{N_2}{N}\frac{k}{N_2v_2(X)}\\ 
    \omega_2 \quad if \quad\frac{N_1}{N}\frac{k}{N_1v_1(X)} < \frac{N_2}{N}\frac{k}{N_2v_2(X)}\end{cases}$$</p>
So:
<p style="text-align: center;">$$g(x) = \begin{cases} \omega_1 \quad if \quad d(X,Knn_2) > d(X,Knn_1)\\ 
    \omega_2 \quad if \quad  d(X,Knn_2) < d(X,Knn_1)  \end{cases}$$</p>

where Knn_i is the k-nearest-neighbor of i^t^h class
Since v(X) is a shere, its volume is proportional to the radius so:

<p style="text-align: center;">$$g(x) = \begin{cases} \omega_1 \quad if \quad v_2(X) > v_1(X)\\ 
\omega_2 \quad if \quad v_2(X)< v_1(X)  \end{cases}$$</p>

For documents, cosine similarity is better than correlation or euclidean.

Data preprocessing is required: attributes may have to be scaled to prevent distance measures from being dominated by one attribute, time series are often standardized to have 0 mean and 1 sd.

Choosing k, if too small sensitive to noise points, if too large neighborhood made of other classes.

NN classifiers are local classifiers, they can produce decision boundaries of arbitrary shape.

Missing values: some approaches use the subset of attributes present in two instances (proximities are not comparable).

Irrelevant attributes add noise to the proximity measure, redundant attributes bias the proximity measure towards certain attribute.

Memory based learner, lazy because doesn't create a synthetic model, easy to implement, no training cost but for bigger models the cost of classification is quite high.


Improving KNN efficiency:

- Avoiding having to compute distance to all objects in the training set: Multi-dimensional access methods(k-d trees), Fast Approximate similarity search, locality sensitive hashing (LSH);
- Condensing: determine a smaller set of objects that five the same performance;
- Editing: remove objects to improve efficiency.
[back](./)
