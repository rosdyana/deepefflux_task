const async = require('async');
const fs = require('fs');
const moment = require('moment');
const http = require('http');
const https = require('https');

var db = require('knex')({
  client: 'mysql',
  connection: {
    host : '140.138.155.216',
    user : 'root',
    password : 'bio1607b',
    database : 'deepefflux'
  }
});

var config = require('./config.json');

var server = require('http').createServer();
var io = require('socket.io')(server);
io.on('connection', function(client){
  console.log('User connected!');
  client.on('disconnect', function(){
    console.log('User disconnected!');
  });
});
server.listen(config.socket_port);

function predict(id,next){
  console.log('>> Start predicting protein #' + id);
  async.series({
    get_data: function(callback) {
      process.stdout.write('> Checking DB:\t\t\t');
      db('proteins').where('id',id).select('name','data','class_a','class_b','class_c','type').first().then(function (fasta) {
        process.stdout.write('Done\n');
        process.stdout.write('> Protein name:\t\t\t'+fasta.name + '\n');
        if (fasta.class_a !=='' && fasta.class_b !='' && fasta.class_c !='') {
          process.stdout.write('> Proteins already predicted.\n> Familia A: ' + fasta.class_a);
          process.stdout.write('> Proteins already predicted.\n> Familia B: ' + fasta.class_b);
          process.stdout.write('> Proteins already predicted.\n> Familia C: ' + fasta.class_c);
          callback(1);
        } else {
          process.stdout.write('> Preparing for conversion:\t');
          fs.writeFile(config.root_path + 'data.fasta',fasta.name + '\n' + fasta.data,(err) => {
            process.stdout.write('Done\n');
            callback();
          })
        }
      });
    }
	,convert: function(callback) {
      process.stdout.write('> Converting to PSSM:\t\t');
      var ps = require('child_process').spawn(config.blast_path,[
        '-db', config.blast_db_path,
        '-num_iterations', 2,
        '-in_msa', config.root_path + 'data.fasta',
        '-out_ascii_pssm', config.root_path + 'data.pssm'
      ]);
      ps.stdout.on('data', (data) => {
        if (process.argv.indexOf('-v')!==-1)
        console.log(`${data}`);
      });
      ps.on('close', (code) => {
       if (code) return callback(code);
        process.stdout.write('Done\n');
        callback();
      });
    },
	calculate: function(callback) {
      process.stdout.write('> Extracting feature PSSM:\t');
      var ps = require('child_process').spawn(config.python_path,[
        config.root_path + 'calculate.py',
		config.root_path + 'data.pssm',
		config.root_path + 'data.csv'
      ]);
      ps.stdout.on('data', (data) => {
        if (process.argv.indexOf('-v')!==-1)
        console.log(`${data}`);
      });
      ps.on('close', (code) => {
    if (code) return callback(code);
        process.stdout.write('Done\n');
        callback();
      });
    },
	predict: function(callback) {
      process.stdout.write('> Predicting:\t\t\t');
      var ps = require('child_process').spawn(config.python_path,[
        config.root_path + 'model.py',
        config.root_path + 'data.csv',
        config.root_path + 'data.out'
      ]);
      ps.stdout.on('data', (data) => {
        if (process.argv.indexOf('-v')!==-1)
        console.log(`${data}`);
      });
      ps.stderr.on('data', (data) => {
        if (process.argv.indexOf('-v')!==-1)
        console.log(`${data}`);
      });
      ps.on('close', (code) => {
        if (code) return callback(code);
        process.stdout.write('Done\n');
        callback();
      });
    },
	updateDB: function(callback){
      fs.readFile(config.root_path + 'data.out','utf8',function (err,data) {
        if (err)  return callback(err); else {
          var result = data.split(",")
          var familia_a = result[0];
          var familia_b = result[1];
          var familia_c = result[2];
          var type_of_familia = result[3];
          process.stdout.write('> data:\t\t\t' + data + '\n');
          process.stdout.write('> class_a:\t\t\t' + familia_a + '\n');
          process.stdout.write('> class_b:\t\t\t '+ familia_b + '\n');
          process.stdout.write('> class_c:\t\t\t '+ familia_c + '\n');
          process.stdout.write('> type:\t\t\t '+ type_of_familia + '\n');
          process.stdout.write('> Saving:\t\t\t');
          db('proteins').where('id',id).update({class_a:familia_a, class_b:familia_b, class_c:familia_c, type:type_of_familia}).then(function () {
            async.each(['data.fasta','data.pssm','data.csv','data.out','error.log'],function(file, call) {
             fs.unlink(config.root_path + file,function () { call() })
            }, function(err) {
             console.log('Done\n')
              callback();
            })
          });
        }
      });
    }
  },
  function(err, results) {
    if (err) return next(err);
    next();
  });
}

Array.prototype.clean = function(deleteValue) {
  for (var i = 0; i < this.length; i++) {
    if (this[i] == deleteValue) {
      this.splice(i, 1);
      i--;
    }
  }
  return this;
};


function doTask(task,next) {
  var start = moment().format("YYYY-MM-DD HH:mm:ss");;
  console.log('\n\n>>> START PROCESSING TASK #' + task.id +'\n');
    async.eachSeries(task.proteins.split(','),function (id,callback) {
        if (id!=='')
            predict(id,()=>{callback()});
        else callback();
    }, function (err) {
      var end = moment().format("YYYY-MM-DD HH:mm:ss");;
      db('tasks').where('id',task.id).update({status:'done', start_time: start, finish_time: end }).then(function () {
        process.stdout.write('>>> FINISH TASK #' + task.id +'\n');
        http.get(config.web + task.id, function(res) {
          var rs = '';
          res.on('data', (data) => {
            rs+=data;
          });
          res.on('end',()=>{
            if (rs=='ok') process.stdout.write('>>> SEND EMAIL.\n\n');
            next();
          });
        });
      });
    });
}

process.stdout.write('Checking new job...');
var loop = setInterval(checkJob, config.check_delay);
function checkJob() {
  db('tasks').where('status','queue').orderBy('submit_time').select().first().then(function (task) {
    if (typeof task == 'undefined') {
      process.stdout.write('.');
    } else {
      clearInterval(loop);
      doTask(task,()=> {
        process.stdout.write('Checking new job...');
        loop = setInterval(checkJob, config.check_delay);
      });
    }
  })
}
