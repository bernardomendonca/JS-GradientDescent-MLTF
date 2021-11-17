const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(feaures, labels, options) {
    this.features = features;
    this.labels = labels;

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
    const currentGuessesForMPG = this.features.map((row) => {
      return this.m * row[0] + this.b;
    });

    // Slope of Mean Square Error in respect to b
    // (2/n) * Summation of ((mx - b) - Actual value)
    // n = total numbers of observations
    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          // our guess minus the actual value of Miles per Gallon
          return guess - this.labels[i][0];
        })
      ) *
        2) /
      this.features.length;

    // Slope of Mean Square Error in respect to m
    // (2/n) * Summation of -x(Actual value - (mx+b))
    // n = total numbers of observations
    const mSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })
      ) *
        2) /
      this.features.length;

    // updating our values with m and b
    // (slope * learning rate) and subtract b and m
    this.m = this.m - mSlope * this.options.learningRate;
    this.b = this.b - bSlope * this.options.learningRate;
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
