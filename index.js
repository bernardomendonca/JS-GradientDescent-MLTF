require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regresion");

//load the CSV file
let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  //the options object takes the learning rate and number of iterations
  learningRate: 0.0001,
  iterations: 100,
});

regression.train();

console.log(
  "Updated value of m is:",
  regression.m,
  "Updated value of b is:",
  regression.b
);
