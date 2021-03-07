const parse = require('csv-parse/lib/sync')
const mean = require('online-mean')
const std = require('online-std')
const LabelEncoder = require('@jsmlt/jsmlt/distribution/preprocessing/labelencoder')
const cplot = require('cplot')

let XGB; require('./xgboost-wrapper.js').then((m) => { XGB = m })

function mae (yt, yp) {
  return yt.reduce((a, v, i) => a + Math.abs(v - yp[i]) / yt.length, 0)
}

function acc (yt, yp) {
  return yt.reduce((a, v, i) => a + (v === yp[i]) / yt.length, 0)
}

module.exports = class Process {
  constructor () {
    console.log('Constructor')
    this.file = ''
    this.records = []
    this.keys = []
  }

  run (p) {
    const params = {
      'train': p['Train set'],
      'test': p['Test set'],
      'target': p['Target variable'],
      'model': p['Model type'],
      'iterations': p['Iterations'],
      'max_depth': p['Max depth'],
      'cv': p['Cross-validation']
    }
    if (!params.train && !this.train.length) {
      console.log('[Fit] No file provided')
      throw new Error('No file selected')
    } else if (params.train !== this.train) {
      console.log('[Fit] Parsing the file...')
      this.train = params.train
      this.Rtrain = parse(params.train, {
        columns: true,
        skip_empty_lines: true
      })
      this.keys = Object.keys(this.Rtrain[0]).filter(key => key.length)
      return {
        'Target variable': {
          options: this.keys
        }
      }
    } else {
      const features = this.keys.filter(k => k !== params.target)
      const Xtrain = this.Rtrain.map(row => features.map(f => row[f]))
      const ytrain = this.Rtrain.map(row => row[params.target])

      const objectives = {
        'XGBoost Regressor': 'reg:linear',
        'XGBoost Classifier': 'binary:logistic',
        'XGBoost Survival': 'survival:cox',
        'XGBoost Count': 'count:poisson'
      }
      // const Dtrain = this.Rtrain.map(row => this.keys.map(k => row[k]))

      // Encode features of the training dataset
      const Xencoders = []
      const eXtrain = Xtrain.map(row => row.slice(0))
      for (let ci = 0; ci < features.length; ci++) {
        console.log('Encoding Xtrain:', features[ci])
        const col = eXtrain.map(row => row[ci])
        const na = col.reduce((a, x) => a + isNaN(x), 0)
        let colen
        if (na > eXtrain.length / 2) {
          // Label encode column
          const encoder = new LabelEncoder()
          colen = encoder.encode(col)
          Xencoders.push(encoder)
        } else {
          // Convert to numbers or fill with mean value
          const m = mean()(col)
          colen = col.map(v => !isNaN(v) ? +v : m)
          Xencoders.push(null)
        }
        eXtrain.forEach((row, ri) => { row[ci] = colen[ri] })
      }

      // Encode the target
      let yencoder
      let eytrain
      if (ytrain.reduce((a, x) => a + isNaN(x), 0) > ytrain.length / 2) {
        yencoder = new LabelEncoder()
        eytrain = yencoder.encode(ytrain)
      } else {
        yencoder = null
        eytrain = ytrain.map(v => !isNaN(v) ? +v : null)
      }

      // const ti = this.keys.findIndex(k => k === params.target)
      // const Xtrain = Dtrain.map(row => row.filter((v, i) => i !== ti))
      // const ytrain = Dtrain.map(row => row[ti])

      console.log('Encoded:', Xtrain, ytrain, Xencoders, yencoder)

      const booster = new XGB({
        booster: 'gbtree',
        objective: objectives[params.model],
        max_depth: +params['max_depth'],
        eta: 0.1,
        min_child_weight: 1,
        subsample: 0.5,
        colsample_bytree: 1,
        silent: 0,
        iterations: +params['iterations']
      })
      booster.train(eXtrain, eytrain)

      const mtrain = params.model.includes('Class')
        ? acc(eytrain, booster.predict(eXtrain).map(Math.round)).toFixed(5) + ' (Accuracy)'
        : mae(eytrain, booster.predict(eXtrain)).toFixed(5) + ' (MAE)'

      const res = {
        'Model': {
          'content': JSON.stringify(booster.toJSON()),
          'filename': 'model.json'
        },
        'Train set': mtrain
      }

      if (params.cv) {
        const nfolds = 10
        const mcv = []
        for (let fold = 0; fold < nfolds; fold++) {
          console.log('Running fold', fold)
          const cXtrain = []
          const cXtest = []
          const cytrain = []
          const cytest = []
          eXtrain.forEach((x, ri) => {
            if (Math.random() > 0.2) {
              cXtrain.push(x)
              cytrain.push(eytrain[ri])
            } else {
              cXtest.push(x)
              cytest.push(eytrain[ri])
            }
          })
          const cbooster = new XGB({
            booster: 'gbtree',
            objective: objectives[params.model],
            max_depth: +params['max_depth'],
            eta: 0.1,
            min_child_weight: 1,
            subsample: 0.5,
            colsample_bytree: 1,
            silent: 0,
            iterations: +params['iterations']
          })
          cbooster.train(cXtrain, cytrain)
          const cmtest = params.model.includes('Class')
            ? acc(cytest, cbooster.predict(cXtest).map(Math.round))
            : mae(cytest, cbooster.predict(cXtest))
          mcv.push(cmtest)
          cbooster.free()
        }
        console.log('MCV:', mcv)
        res['CV'] = {
          'content': cplot({
            title: 'CV',
            xAxis: {
              label: 'Fold'
            },
            yAxis: {
              label: 'Value'
            },
            plot: {
              color: '#5700AD',
              marker: 'x',
              x: Array(nfolds).fill(0).map((v, i) => i),
              y: mcv
            },
            width: 600,
            height: 400
          }),
          'filename': 'cv.svg'
        }
        const m = mean()
        const s = std()
        mcv.forEach(v => { m(v); s(v) })
        res['Train set (CV)'] = m().toFixed(5) + ' ±' + (2 * s()).toFixed(5) + (params.model.includes('Class') ? ' (Accuracy)' : ' (MAE)')
      }

      if (params.test) {
        // Predict new data
        const Rtest = parse(params.test, {
          columns: true,
          skip_empty_lines: true
        })

        const Xtest = Rtest.map(row => features.map(f => row[f]))
        const ytest = params.target in Rtest[0]
          ? Rtest.map(row => row[params.target])
          : null

        const eXtest = Xtest.map(row => row.slice(0))
        for (let ci = 0; ci < features.length; ci++) {
          const col = eXtest.map(row => row[ci])
          console.log('Encoding Xtest:', features[ci])
          let colen
          const encoder = Xencoders[ci]
          if (encoder !== null) {
            // Label encode column
            colen = encoder.encode(col)
          } else {
            // Convert to numbers or fill with mean value
            const m = mean()(col)
            colen = col.map(v => !isNaN(v) ? +v : m)
          }
          eXtest.forEach((row, ri) => { row[ci] = colen[ri] })
        }

        let csv = ''
        const _ypred = booster.predict(eXtest)
        console.log('True:', ytest)
        console.log('Raw:', _ypred)

        let mtest
        let plot

        if ((yencoder !== null) || (params.model.includes('Classifier'))) {
          const ypredclass = _ypred.map(Math.round)
          console.log('EEE', ypredclass)
          const ypred = (yencoder === null) ? ypredclass : yencoder.decode(ypredclass)
          console.log('Encoded:', ypred)
          csv = features.join(',') + ',' + params.target + '_prob,' + params.target + '\n'
          csv += Xtest.map((row, ri) => row.join(',') + ',' + _ypred[ri] + ',' + ypred[ri]).join('\n') + '\n'
          mtest = ytest === null ? null : acc(ypred, ytest).toFixed(5) + ' (Accuracy)'
          if (ytest === null) {
            plot = null
          } else {
            const fpr = []
            const tpr = []
            const eytest = (yencoder === null) ? ytest.map(v => +v) : yencoder.encode(ytest)
            const P = eytest.reduce((a, v) => a + v, 0)
            const N = eytest.length - P
            for (let t = 0; t < 1; t += 0.01) {
              let FP = 0
              let TP = 0
              _ypred.forEach((yp, i) => {
                if (yp > t) {
                  if (eytest[i] === 1) {
                    TP += 1
                  } else {
                    FP += 1
                  }
                }
              })
              fpr.push(FP / N)
              tpr.push(TP / P)
            }
            plot = cplot({
              title: 'ROC curve',
              xAxis: {
                label: 'False positive rate'
              },
              yAxis: {
                label: 'True positive rate'
              },
              plots: [
                {
                  color: '#5700AD',
                  marker: 'x',
                  x: fpr,
                  y: tpr
                },
                {
                  color: '#DDD',
                  x: Array(100).fill(0).map((v, i) => i / 100),
                  y: Array(100).fill(0).map((v, i) => i / 100)
                }
              ],
              width: 600,
              height: 600
            })
          }
        } else {
          csv = features.join(',') + ',' + params.target + '\n'
          csv += Xtest.map((row, ri) => row.join(',') + ',' + _ypred[ri]).join('\n') + '\n'
          mtest = ytest === null ? null : mae(_ypred, ytest.map(v => +v)).toFixed(5) + ' (MAE)'
          plot = ytest === null
            ? null
            : cplot({
              title: 'True x Predicted',
              xAxis: {
                label: 'True'
              },
              yAxis: {
                label: 'Predicted'
              },
              scatter: {
                color: '#5700AD',
                marker: 'x',
                x: ytest.map(v => +v),
                y: _ypred
              },
              width: 600,
              height: 600
            })
        }

        res['Prediction'] = {
          'content': csv,
          'filename': 'prediction.csv'
        }

        if (mtest !== null) {
          res['Test set'] = mtest
        }

        if (plot !== null) {
          res['Plot'] = {
            'content': plot,
            'filename': 'plot.svg'
          }
        }
      } // *if test set is present

      return res
    }
  }
}
