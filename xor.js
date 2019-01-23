console.log("Hello");

const model = tf.sequential();
const hidden = tf.layers.dense({
    units: 2,
    inputShape: [2],
    activation: 'tanh'
});
model.add(hidden);

const output = tf.layers.dense({
    units: 1,
    activation: 'tanh'
});
model.add(output);

const sgdOpt = tf.train.adam(0.1)
model.compile({
    optimizer: sgdOpt,
    loss: tf.losses.meanSquaredError
})

const x_train = tf.tensor2d([
    [1,1],
    [1,0],
    [0,1],
    [0,0]
]);

const y_train = tf.tensor2d([
    [0],
    [1],
    [1],
    [0]
]);

train().then( () => {
    console.log("training complete");
    let outputs = model.predict(x_train);
    outputs.print();
    console.log(model);
});
async function train() {
    const config = {
        shuffle: true,
    }
    for(let i = 0; i < 200; i ++){
        const response = await model.fit(x_train, y_train, config);
        console.log(response.history.loss[0])
    }
}