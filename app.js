var app = new Vue({
    el: '#app',
    data() {
        return {
            message: 'new Vue instance',
            collectStack: [],
            desiredArr: []
        }
    },

    mounted() {

        this.init();
        console.log('done')
    },

    methods: {

        async createData() {
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
                this.desiredArr.push({
                    'x1': concattedFinal.gather(indices).get([0]),
                    'x2': concattedFinal.gather(indices).get([1]),
                    'class': concattedFinal.gather(indices).get([2])
                })
            }


        },

        async init() {
            await this.createData();

        },

        createPlot(){

        }
    }

});