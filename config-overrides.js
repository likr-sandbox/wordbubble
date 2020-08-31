module.exports = function override(config, env) {
  config.externals = {
    "@tensorflow/tfjs": "window.tf",
    "@tensorflow/tfjs-vis": "window.tfvis",
  };
  return config;
};
