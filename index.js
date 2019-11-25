const tf = require('@tensorflow/tfjs')
// const data = require('./testdata.js')

const data = []

for(let i = 0; i < 40; ++i) {
    data[i] = {
        cos: Math.cos(i),
        sin: Math.sin(i)
    }
}


const TIME_STEPS = 4
const FEATURE_COUNT = 2

const model = tf.sequential()
model.add(tf.layers.lstm({
  units: 16,
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

model.add(tf.layers.dense({units: 1, activation: 'linear'}))
// model.add(tf.layers.timeDistributed(
//     {layer: tf.layers.dense({units: OUTPUT_VOCABULARY})}));

const optimizer = tf.train.rmsprop(0.1)
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
        buffer.set(value.cos, exampleIndex, sequenceIndex, 0);
        buffer.set(value.sin, exampleIndex, sequenceIndex, 1);
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
        buffer.set(value.cos+value.sin, exampleIndex, 0);
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

model.fit(xs, ys, {epochs: 10}).then(h => {
    console.log('loss', h.history.loss)

    const rando = Math.floor(Math.random() * 100)+ 60
    const randoTest = []
    for(let i = 0; i < 4; ++i) {
        randoTest[i] = {
            cos: Math.cos(rando+i),
            sin: Math.sin(rando+i)
        }
    }
    console.log('predict', model.predict(encodeBatch([randoTest], 4)).arraySync(), Math.cos(rando+4)+Math.sin(rando+4))
})

