#include "common.h"

std::vector<float> squared_pairwise_dist(std::vector<float> &points, const unsigned int N, const unsigned int NDIMS) {
	std::vector<float> squared_pairwise_dist(N * N, 0);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for(int k = 0; k < NDIMS; k++) {
				squared_pairwise_dist[i * N + j] += (points[i * NDIMS + k] - points[j * NDIMS + k]) * (points[i * NDIMS + k] - points[j * NDIMS + k]); 
			}
		}
    }
	return squared_pairwise_dist;
}



bool perplexity_equal(const float delta, float perplexity, float target_perplexity) {
	return (perplexity >= target_perplexity - delta) && (perplexity <= target_perplexity + delta);
}


float get_perplexity(std::vector<float> & pij, const unsigned int i, unsigned int N) {
	float entropy = 0.0f;
	for (int j = 0; j < N; j++) {
		if (j != i) {
			entropy += pij[i*N + j] * std::log2(pij[i*N + j]);
		}
	}
	return std::pow(2, -entropy);
}

bool compare_perplexity(std::vector<float>& pij, 
					   float& lo, 
					   float& mid, 
					   float& hi, 
					   const unsigned int i, 
					   const unsigned int N, 
					   const float delta, 
					   const float target_perplexity) {
	float perplexity = get_perplexity(pij, i, N);
	if (perplexity_equal(delta, perplexity, target_perplexity)) {
		return true;
	} else if (perplexity > target_perplexity) {
		hi = mid - delta;
	} else {
		lo = mid + delta;
	}

	mid = (lo + hi)/2;
	return false;
}


void recompute_pij_row_cpu(std::vector<float> &points, 
	                           std::vector<float> &pij, 
	                           float sigma,
	                           float i, 
	                           const unsigned int N, 
	                           const unsigned int NDIMS) {
	std::vector<float> dists = squared_pairwise_dist(points, N, NDIMS);
	for (int j = 0; j < N; j++) {
		float denom = 0;
		for (int k = 0; k < N; k++) {
			if (k != i) {
				denom += std::exp(-(dists[i * N + k] / (2 * sigma * sigma)));
			}
		}
        if (i != j) {
		    pij[i * N + j] = std::exp(-dists[i * N + j] / (2 * sigma * sigma)) / denom;
        }

    }
}

std::vector<float> compute_pij_cpu(std::vector<float> &points, 
	                           std::vector<float> &sigma, 
	                           const unsigned int N, 
	                           const unsigned int NDIMS) {
	std::vector<float> pij_out(N * N, 0.0f);
	std::vector<float> dists = squared_pairwise_dist(points, N, NDIMS);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float denom = 0;
			for (int k = 0; k < N; k++) {
				if (k != i) {
					denom += std::exp(-(dists[i * N + k] / (2 * sigma[i] * sigma[i])));
				}
			}
            if (i != j) {
			    pij_out[i * N + j] = std::exp(-dists[i * N + j] / (2 * sigma[i] * sigma[i])) / denom;
            }
           
        }
	}
	return pij_out;

}

std::vector<float> compute_qij_cpu(std::vector<float>& ys, const unsigned int N, const unsigned int PROJDIMS) {
	
	std::vector<float> qij_out(N * N);
	std::vector<float> dists = squared_pairwise_dist(ys, N, PROJDIMS);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float denom = 0;
			for (int k = 0; k < N; k++) {
				if (k != i) {
					denom += 1 / (1 + (dists[i * N + k]));
				}
			}
			qij_out[i * N + j] = (1 / (1 + dists[i * N + j])) / denom;
		}
	}
	return qij_out;
}

float kl_div(float pij, float qij) {
	return pij * std::log(pij / qij);
}

float compute_gradients_cpu(std::vector<float> &forces,
                    	std::vector<float> &dist, 
                        std::vector<float> &ys, 
                        std::vector<float> &pij, 
                        std::vector<float> &qij, 
                        const unsigned int N,
                        float eta) {

	float loss = 0.0f;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float pij_sym = (pij[i * N + j] + pij[j * N + i]) / (2 * N);
			float qij_sym = (qij[i * N + j] + qij[j * N + i]) / (2 * N);
			loss += kl_div(pij_sym, qij_sym);
		}

	}
	return loss;

}


std::vector<float> sigmas_search_cpu(std::vector<float> &points,  
	                           const unsigned int N, 
	                           const unsigned int NDIMS,
	                           float target_perplexity) {
	const float max_sigma = 1000.0f;
	const float delta = 0.1f;
	std::vector<float> sigmas(N, max_sigma/2);
	std::vector<float> pij = compute_pij_cpu(points, sigmas, N, NDIMS);
	for (int i = 0; i < N; i++) {
		bool found = false;
		float lo = 0.0f;
		float hi = max_sigma;
		float mid = (lo + hi)/ 2;
		while (!found) {
			found = compare_perplexity(pij, lo, mid, hi, i, N, delta, target_perplexity);
			recompute_pij_row_cpu(points, pij, mid, i, N, NDIMS);
        }
		sigmas[i] = mid;
	}
	return sigmas;

}

std::vector<float> naive_tsne_cpu(std::vector<float> &points, 
                              const unsigned int N, 
                              const unsigned int NDIMS) {
	std::default_random_engine generator;
  	std::uniform_real_distribution<double> distribution(-10.0f,10.0f);
 	const unsigned int NPROJDIM = 2;
 	std::vector<float> ys(N * NPROJDIM);
 	for (int i = 0; i < N * NPROJDIM; i++) {
 		ys[i] = distribution(generator);
 	}
	for (int i = 0; i < 1000; i++) {

	}
    return ys;

}
