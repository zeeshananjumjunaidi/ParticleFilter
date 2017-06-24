/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *	Updated on: Jun 24, 2017
 *		Author: Zeeshan Anjum
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <numeric>
#include "particle_filter.h"

using namespace std;
#define NUMBER_OF_PARTICLES 50 // No of Particles
#define EPS 0.001 // small number

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	static default_random_engine gen;
	gen.seed(123);
	num_particles = NUMBER_OF_PARTICLES; // init number of particles to use
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	particles.resize(num_particles); // Resize the particles vector to fit desired number of particles
	weights.resize(num_particles);
	double init_weight = 1.0 / num_particles;
	for (int i = 0; i < num_particles; i++) {
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = init_weight;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// Some constants to save computation power
	const double vel_d_t = velocity * delta_t;
	const double yaw_d_t = yaw_rate * delta_t;
	const double vel_yaw = velocity / yaw_rate;
	static default_random_engine gen;
	gen.seed(571);
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < EPS) {
			// Motion Equation for X' = X + V * dt * cos(Theta)
			particles[i].x += vel_d_t * cos(particles[i].theta);
			// Motion Equation for Y' = Y + V * dt * sin(Theta)
			particles[i].y += vel_d_t * sin(particles[i].theta);
			// Angle Theta remains unchanged if yaw_rate is smaller than threshold
		}
		// if theta is not zero (near to zero)
		else {
			double new_theta = particles[i].theta + yaw_d_t;
			// x​f​​=x​0​​+​​θ​˙​​​​v​​[sin(θ​0​​+​θ​˙​​(dt))−sin(θ​0​​)]
			particles[i].x += vel_yaw * (sin(new_theta) - sin(particles[i].theta));
			// y​f​​=y​0​​+​​θ​˙​​​​v​​[cos(θ​0​​)−cos(θ​0​​+​θ​˙​​(dt))]
			particles[i].y += vel_yaw * (-cos(new_theta) + cos(particles[i].theta));
			// θ​f​​=θ​0​​+​θ​˙​​(dt)
			particles[i].theta = new_theta;
		}
		// Add random Gaussian noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (size_t i = 0; i < observations.size(); i++)
	{
		//Set  the distance as maximum
		if (observations.size() > 0) {
			double dist_x = observations[0].x;
			double dist_y = observations[0].y;
			for (size_t j = 0; j < predicted.size(); j++)
			{
				double x_ = predicted[j].x;
				double y_ = predicted[j].y;
				double xO_ = observations[i].x;
				double yO_ = observations[i].y;
				if (abs(x_ - xO_) < dist_x && abs(y_ - yO_) < dist_y) {
					dist_x = x_;
					dist_y = y_;
				}
			}
			observations[i].x = dist_x;
			observations[i].y = dist_y;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	double sigma_xx = std_landmark[0] * std_landmark[0];
	double sigma_yy = std_landmark[1] * std_landmark[1];
	double norm_weight = 1 / sqrt(2 * M_PI * std_landmark[0] * std_landmark[1]);
	double dx = 0.0;
	double dy = 0.0;
	double sum_w = 0.0; // Sum of all weights, used for normalizing weights
	for (int i = 0; i < num_particles; i++) {
		double weight_without_exp = 0.0;
		double sin_theta = sin(particles[i].theta);
		double cos_theta = cos(particles[i].theta);

		//dataAssociation(observations, observations);
		for (int j = 0; j < observations.size(); j++) {
			// Observation measurement transformations
			LandmarkObs observation;
			observation.id = observations[j].id;
			observation.x = particles[i].x + (observations[j].x * cos_theta) - (observations[j].y * sin_theta);
			observation.y = particles[i].y + (observations[j].x * sin_theta) + (observations[j].y * cos_theta);


			bool in_range = false;
			Map::single_landmark_s nearest_lm;
			double nearest_dist = 10000000.0; // A big number
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s cond_lm = map_landmarks.landmark_list[k];
				// Calculate the Euclidean distance between two 2D points
				double distance = dist(cond_lm.x_f, cond_lm.y_f, observation.x, observation.y);
				if (distance < nearest_dist) {
					nearest_dist = distance;
					nearest_lm = cond_lm;
					if (distance < sensor_range) {
						in_range = true;
					}
				}
			}
			if (in_range)
			{
				// xi-ui from the weight formula
				dx = observation.x - nearest_lm.x_f;
				dy = observation.y - nearest_lm.y_f;
				// we won't calclate the same calulcation for the weight on each iteration
				// rather than we calculate only the changing variables such as x and mu
				// we will apply exponent and normalizing term after iteration on observations.
				weight_without_exp += pow(dx, 2) / sigma_xx + pow(dy,2) / sigma_yy;
			}
			else {
				//After multiplying with -0.5 and applying exponent it will reduced to smallest value
				weight_without_exp += 100;
			}
		}
		// here we only calculate exp(-1/2*(xi-ui)T*SigmaInv*(xi-ui)
		particles[i].weight = exp(-0.5*weight_without_exp);
		sum_w += particles[i].weight;
	}
	// Weights normalization to sum(weights)=1
	for (int i = 0; i < num_particles; i++) {
		// here we normalize the weight of each particle by dividing by total
		// of weights.
		// then we apply the constant term 1/sqrt(2PI*Sigma) for each particle weight
		particles[i].weight /= sum_w * norm_weight;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	static default_random_engine gen;
	// seed the random number generator
	gen.seed(123);
	discrete_distribution<> dist_particles(weights.begin(), weights.end());
	vector<Particle> new_particles;
	new_particles.resize(num_particles);
	for (int i = 0; i < num_particles; i++) {
		new_particles[i] = particles[dist_particles(gen)];
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
