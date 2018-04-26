const m = 400;
const N = m / 2;
const D = 2;

const a = tf.scalar(4);

const z = tf.variable(tf.scalar(32));
var collectStack = []

for (var j = 0; j < 2; j++) {

    const ix = tf.range(N * j, N * (j + 1))
    const t = tf.add(tf.linspace(j * 3.12, (j + 1) * 3.12, N), tf.randomNormal([N]).mul(tf.scalar(0.2)))
    const r = tf.add(tf.mul(a, tf.sin(tf.scalar(4).mul(t))), tf.randomNormal([N]).mul(tf.scalar(0.2)));
    console.log(`The shape of r is ${r.shape}`)
    const last = tf.mul(r, tf.sin(t)).concat(tf.mul(r, tf.cos(t)))
    const sinned = tf.mul(r, tf.sin(t));

    console.log(`The shape of sinned is ${sinned.shape}`)
    const cossed = tf.mul(r, tf.cos(t));

    const stackSinCos = tf.stack([sinned, cossed], 1)

    console.log(`The shape of stacksincos is ${stackSinCos.shape}`)

    this.collectStack.push(stackSinCos)

    console.log(`The shape of cossed is ${cossed.shape}`)
}

const concatted = tf.concat([this.collectStack[0], this.collectStack[1]]);
const Ycreated = tf.fill([200, 1], 1).concat(tf.fill([200, 1], 0))

const concattedFinal = tf.concat([concatted, Ycreated], 1)

const desiredArr = []


for (var i = 0; i < 400; i++) {
    const indices = tf.tensor1d([i])
    desiredArr.push({
        'x1': concattedFinal.gather(indices).get([0]),
        'x2': concattedFinal.gather(indices).get([1]),
        'class': concattedFinal.gather(indices).get([2])
    })
}


var app = new Vue({
    el: '#app',
    data() {
        return {
            accuracyLR: '',
            accuracyNeu: null,
            plotShown: false,
            X1: null,
            X2: null,
            prob: null,
            predictMessage: '',
            buttonText: 'Train with Logistic Regression',
            showNetButton: false,
        }
    },

    mounted() {

        this.createPlot()
    },

    methods: {



        createPlot() {

            const margin = {
                top: 20,
                right: 20,
                bottom: 30,
                left: 40
            };

            const width = 500 - margin.left - margin.right;
            const height = 450 - margin.top - margin.bottom;

            const xValue = d => d.x1
            const xScale = d3.scaleLinear().range([0, width])

            const xMap = d => xScale(xValue(d))
            const xAxis = d3.axisBottom(xScale)

            const yValue = d => d.x2
            const yScale = d3.scaleLinear().range([height, 0])

            const yMap = d => yScale(yValue(d))
            const yAxis = d3.axisLeft(yScale)

            // const cValue = d => d.Manufacturer
            const cValue = d => d.class
            const color = d3.scaleOrdinal(d3.schemeCategory10);


            // xScale.domain([d3.min(parsed, xValue) - 1, d3.max(parsed, xValue) + 1])
            xScale.domain([d3.min(desiredArr, xValue) - 1, d3.max(desiredArr, xValue) + 1])
            // yScale.domain([d3.min(parsed, yValue) - 1, d3.max(parsed, yValue) + 1])
            yScale.domain([d3.min(desiredArr, yValue) - 1, d3.max(desiredArr, yValue) + 1])

            const svg = d3.select('#plotSection').append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', width + margin.top + margin.bottom)
                .append('g')
                .attr('transform', `translate(${margin.left},${margin.top})`)

            const tooltip = d3.select('#plotSection').append('div')
                .attr('class', 'tooltip')
                .style('opacity', 0)

            svg.append('g')
                .attr('class', 'x axis')
                .attr('transform', `translate(0,${height})`)
                .call(xAxis)
                .append('text')
                .attr('class', 'axis')
                .attr('x', width)
                .attr('y', -6)
                .attr('fill', '#000')
                .style('text-anchor', 'end')
                .text('X1')

            svg.append('text')
                .attr('x', width / 4)
                .attr('y', height + 40)
                .attr('fill', '#000')
                .attr('style', 'bold')
                .text('A scatterplot of two variables')


            svg.append('g')
                .attr('class', 'y axis')
                .call(yAxis)
                .append('text')
                .attr('class', 'axis')
                .attr('transform', 'rotate(-90)')
                .attr('y', 6)
                .attr('dy', '.71em')
                .attr('fill', '#000')
                .style('text-anchor', 'end')
                .text('X2')

            svg.selectAll('.dot')
            // .data(parsed)
                .data(desiredArr)
                .enter().append('circle')
                .attr('class', 'dot')
                .attr('r', 6.5)
                .attr('cx', xMap)
                .attr('cy', yMap)
                // .style('fill', function(d) { return color(cValue(d))})
                .style('fill', d => color(cValue(d)))
                .on('mouseover', d => {
                    tooltip.transition()
                        .duration(200)
                        .style('opacity', 0.9);
                    tooltip.html(`Class: ${d.class}<br/>X1: ${xValue(d).toFixed(2)}, X2: ${yValue(d).toFixed(2)}`)
                    // .style('left', `${d3.event.pageX + 5}px`)
                        .style('left', `${d3.event.pageX - 210}px`)
                        // .style('top', `${d3.event.pageY -28 }px`)
                        .style('top', `${d3.event.pageY }px`)
                        .style('border', '1px solid grey')
                        .style('padding-left', '15px')
                        .style('padding-top', '5px')
                        .style('padding-bottom', '5px')
                        .style('background-color', 'lightblue')
                        .style('width', `${d.class.length + 300}px`)
                })
                .on('mouseout', d => {
                    tooltip.transition()
                        .duration(500)
                        .style('opacity', 0)
                })

            var legend = svg.selectAll('.legend')
                .data(color.domain())
                .enter().append('g')
                .attr('class', 'legend')
                .attr('transform', (d, i) => {
                    return `translate(0,${i * 20})`
                })

            legend.append('rect')
                .attr('x', width - 18)
                .attr('width', 18)
                .attr('height', 18)
                .attr('fill', color)


            legend.append('text')
                .attr('x', width - 24)
                .attr('y', 9)
                .attr('dy', '0.35em')
                .attr('text-anchor', 'end')
                .text(d => d)


        },


        makePrediction() {
            console.log(this.$el.textContent[0]);

            const model = tf.sequential();


            this.accuracyLR = 'Training...'

            setTimeout(() => {

                model.add(tf.layers.dense({units: 1, inputShape: [2], activation: 'sigmoid'}))
                model.compile({loss: 'binaryCrossentropy', optimizer: 'sgd'});
                this.fitModel(model)
            }, 100)


        },

        fitModel(model) {
            this.accuracyLR = 'Training ..';
            model.fit(concatted, Ycreated, {epochs: 100}).then(() => {
                const prediction1 = model.predict(concatted)
                const prediction = tf.greater(prediction1, tf.scalar(0.7)).toFloat();
                const accuracy = prediction.equal(Ycreated).sum().get([0]) / 4
                this.accuracyLR = `${accuracy}%`

            })
        },


        trainNeuralNet() {


            this.accuracyNeu = 'Hold On. This may take a few minutes ...'

            const model2 = tf.sequential();

            model2.add(tf.layers.dense({units: 4, inputShape: [2], activation: 'tanh'}))
            model2.add(tf.layers.dense({units: 1, activation: 'sigmoid'}))
            model2.compile({loss: 'binaryCrossentropy', optimizer: 'adam'})

            setTimeout(() => {
                model2.fit(concatted, Ycreated, {epochs: 100}).then(() => {
                    const prediction1 = model2.predict(concatted)
                    const prediction = tf.greater(prediction1, tf.scalar(0.5)).toFloat();
                    const accuracy = prediction.equal(Ycreated).sum().get([0]) / 4
                    // console.log(tf.div(numer, denom).get([0]).toFixed(2))
                    this.accuracyNeu = `${accuracy}%`
                })
            }, 100)


        }
    }

});


