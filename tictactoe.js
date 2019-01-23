const x_train = tf.tensor3d([
    [[0,1,0],[1,0,0],[0,0,1], 
     [0,1,0],[1,0,0],[1,0,0], 
     [1,0,0],[1,0,0],[1,0,0]],
    [[0,1,0],[0,1,0],[1,0,0], 
     [1,0,0],[0,0,1],[1,0,0], 
     [1,0,0],[1,0,0],[1,0,0]],
    [[0,1,0],[1,0,0],[1,0,0], 
     [1,0,0],[0,0,1],[1,0,0], 
     [0,0,1],[1,0,0],[0,1,0]],
    [[0,1,0],[0,0,1],[1,0,0], 
     [0,0,1],[0,1,0],[1,0,0], 
     [1,0,0],[1,0,0],[1,0,0]],
    [[0,1,0],[0,0,1],[0,1,0], 
     [1,0,0],[1,0,0],[1,0,0], 
     [1,0,0],[0,0,1],[1,0,0]],
    [[0,1,0],[0,0,1],[0,1,0], 
     [1,0,0],[1,0,0],[0,0,1], 
     [1,0,0],[1,0,0],[1,0,0]],
    [[0,1,0],[1,0,0],[1,0,0], 
     [1,0,0],[1,0,0],[0,0,1], 
     [1,0,0],[1,0,0],[1,0,0]]
]);

console.log(x_train);

const y_train = tf.tensor2d([
    [0,0,1, 
     0,0,0, 
     0,0,0],
    [0,0,1, 
     0,0,0, 
     0,0,0],
    [0,0,1, 
     0,0,0, 
     0,0,0],
    [0,0,0, 
     0,0,0, 
     0,0,1],
    [0,0,0, 
     0,1,0, 
     0,0,0],
    [0,0,1, 
     0,0,0, 
     0,0,0],
    [0,0,1, 
     0,0,0, 
     0,0,0]
]);
console.log(y_train);
console.log("Hello");

const model = tf.sequential();

const hidden = tf.layers.dense({
    units: 9,
    inputShape: [[9,3]],
    activation: 'tanh'
});
model.add(hidden);

const hidden2 = tf.layers.dense({
    units: 9,
    activation: 'tanh'
});
model.add(hidden2);

const hideen3 = tf.layers.dense({
    units: 9,
    activation: 'tanh'
});
model.add(hideen3);

const output = tf.layers.dense({
    units: 9,
    activation: 'tanh'
});
model.add(output);

const sgdOpt = tf.train.sgd(0.2)
model.compile({
    optimizer: sgdOpt,
    loss: tf.losses.meanSquaredError
});


train().then( () => {
    console.log("training complete");
    let outputs = model.predict(x_train);
    outputs.print();
    console.log(model);
});
async function train() {
    const config = {
        shuffle: true,
        // epoch: 10,
    }
    for(let i = 0; i < 1000; i ++){
        const response = await model.fit(x_train, y_train, config);
        console.log(response.history.loss[0]);
    }
}