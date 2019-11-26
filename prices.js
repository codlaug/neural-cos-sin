const tf = require('@tensorflow/tfjs')
const { MinMaxScaler } = require('machinelearn/preprocessing')

let data = require('./testdata.js').slice(0, 100)


for(let i = 0; i < data.length; ++i) {
  data[i] = [
    data[i].open,
    data[i].close,
    data[i].high,
    data[i].low,
    data[i].volume,
  ]
}


const scaler = new MinMaxScaler({ featureRange: [0, 1] })

//normalize the data
data = scaler.fit_transform(data)


const TIME_STEPS = 4
const FEATURE_COUNT = 5

const model = tf.sequential()
model.add(tf.layers.lstm({
  units: 128,
  inputShape: [TIME_STEPS, FEATURE_COUNT],
//   recurrentInitializer: 'glorotNormal',
//   returnSequences: true
}))
// model.add(tf.layers.repeatVector({n: TIME_STEPS + 1}));
// model.add(tf.layers.lstm({
//     units: 16,
//     recurrentInitializer: 'glorotNormal',
//     // returnSequences: true
//   }))
model.add(tf.layers.dropout(0.5))
model.add(tf.layers.dense({units: 24, activation: 'relu'}))
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}))
// model.add(tf.layers.timeDistributed(
//     {layer: tf.layers.dense({units: OUTPUT_VOCABULARY})}));

const optimizer = tf.train.rmsprop(0.01)
model.compile({optimizer, loss: tf.losses.meanSquaredError})


function encodeBatch(sequences, numRows) {
    const numExamples = sequences.length;
    const buffer = tf.buffer([numExamples, numRows, FEATURE_COUNT]);

    for (let n = 0; n < numExamples; ++n) {
        const exampleIndex = n
      const sequence = sequences[n];
      for (let i = 0; i < sequence.length; ++i) {
          const sequenceIndex = i
        const value = sequence[i];
        buffer.set(value[0], exampleIndex, sequenceIndex, 0);
        buffer.set(value[1], exampleIndex, sequenceIndex, 1);
        buffer.set(value[2], exampleIndex, sequenceIndex, 2);
        buffer.set(value[3], exampleIndex, sequenceIndex, 3);
        buffer.set(value[4], exampleIndex, sequenceIndex, 4);
      }
    }
    return buffer.toTensor().as3D(numExamples, numRows, FEATURE_COUNT);
}

function encodeAnswerBatch(nextChars, numRows) {
    const numExamples = nextChars.length;
    const buffer = tf.buffer([numExamples, 1]);

    for (let n = 0; n < numExamples; ++n) {
        const exampleIndex = n
      const value = nextChars[n];
        buffer.set(value[1], exampleIndex, 0);
    }
    return buffer.toTensor().as2D(numExamples, 1);
}


const values = []
const inputSequences = []
const nextCharsInSequence = []
for(let i = 0; i < data.length-TIME_STEPS-1; ++i) {
  inputSequences.push(data.slice(i, i+TIME_STEPS))
  nextCharsInSequence.push(data[i+TIME_STEPS])
}
// console.log(inputValues)
// const xs = tf.tensor3d(inputValues, [TIME_STEPS, 1, OUTPUT_VOCABULARY])
const xs = encodeBatch(inputSequences, TIME_STEPS)

// const ys = tf.tensor2d(data.slice(1, 11), [1, OUTPUT_VOCABULARY])
const ys = encodeAnswerBatch(nextCharsInSequence, TIME_STEPS)

console.log('xs', xs.arraySync())
console.log('ys', ys.arraySync())

const fitOptions = {
    epochs: 10,
    validationSplit: 0.1,
    callbacks: {
        onEpochEnd: (t,h) => {
            console.log(t,h)
        }
    }
}

model.fit(xs, ys, fitOptions).then(h => {
    console.log('loss', h)

    const rando = Math.floor(Math.random() * data.length-5)
    const randoTest = data.slice(rando, rando+4)
    console.log('predict', scaler.inverse_transform([0, model.predict(encodeBatch([randoTest], 4)).arraySync()[0][0]])[1], scaler.inverse_transform(data[rando+4])[1])
})

