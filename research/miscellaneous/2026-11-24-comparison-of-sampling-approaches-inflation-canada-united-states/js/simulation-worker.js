'use strict';
importScripts('sampling-engine.js');
self.onmessage = event => {
  try {
    const {population, config} = event.data;
    self.postMessage({result:CPISampling.runPopulation(population, config)});
  }
  catch (error) { self.postMessage({error:error.message}); }
};
