using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    public sealed class FlowMatchEulerDiscreteScheduler : SchedulerBase
    {
        private float[] _sigmas;
        private float _sigmaMin;
        private float _sigmaMax;
        private float _shift = 3.0f;

        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDiscreteScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public FlowMatchEulerDiscreteScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDiscreteScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public FlowMatchEulerDiscreteScheduler(SchedulerOptions schedulerOptions) : base(schedulerOptions) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            var timesteps = ArrayHelpers.Linspace(1, Options.TrainTimesteps, Options.TrainTimesteps).Reverse();
            var sigmas = timesteps.Select(x => x / Options.TrainTimesteps);
            sigmas = sigmas.Select(sigma => _shift * sigma / (1f + (_shift - 1f) * sigma)).ToArray();
            timesteps = sigmas.Select(sigma => sigma * Options.TrainTimesteps).ToArray();

            _sigmas = sigmas.ToArray();
            _sigmaMin = _sigmas[^1];
            _sigmaMax = _sigmas[0];
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            // Create timesteps based on the specified strategy
            var timesteps = ArrayHelpers.Linspace(SigmaToT(_sigmaMin), SigmaToT(_sigmaMax), Options.InferenceSteps).Reverse();
            var sigmas = timesteps.Select(x => x / Options.TrainTimesteps);
            sigmas = sigmas.Select(sigma => _shift * sigma / (1 + (_shift - 1) * sigma));
            _sigmas = sigmas.Append(0f).ToArray();
            timesteps = sigmas.Select(sigma => sigma * Options.TrainTimesteps).ToArray();
            return timesteps
                .Select(x => (int)x)
                .OrderByDescending(x => x)
                .ToArray();
        }


        private float SigmaToT(float sigma)
        {
            return sigma * Options.TrainTimesteps;
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public override DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            return sample;
        }


        /// <summary>
        /// Processes a inference step for the specified model output.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="order">The order.</param>
        /// <returns></returns>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            // TODO: Implement "extended settings for scheduler types"
            float s_churn = 0f;
            float s_tmin = 0f;
            float s_tmax = float.PositiveInfinity;
            float s_noise = 1f;

            var stepIndex = Timesteps.IndexOf(timestep);
            float sigma = _sigmas[stepIndex];

            float gamma = s_tmin <= sigma && sigma <= s_tmax ? (float)Math.Min(s_churn / (_sigmas.Length - 1f), Math.Sqrt(2.0f) - 1.0f) : 0f;
            var noise = CreateRandomSample(modelOutput.Dimensions);
            var epsilon = noise.MultiplyTensorByFloat(s_noise);
            float sigmaHat = sigma * (1.0f + gamma);

            if (gamma > 0)
                sample = sample.AddTensors(epsilon.MultiplyTensorByFloat((float)Math.Sqrt(Math.Pow(sigmaHat, 2f) - Math.Pow(sigma, 2f))));

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var denoised = sample.SubtractTensors(modelOutput.MultiplyTensorByFloat(sigmaHat));

            // 2. Convert to an ODE derivative
            var derivative = sample.SubtractTensors(denoised).DivideTensorByFloat(sigmaHat);

            var delta = _sigmas[stepIndex + 1] - sigmaHat;
            return new SchedulerStepResult(sample.AddTensors(derivative.MultiplyTensorByFloat(delta)));
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="originalSamples">The original samples.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        public override DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, IReadOnlyList<int> timesteps)
        {
            var sigma = timesteps
                .Select(x => Timesteps.IndexOf(x))
                .Select(x => _sigmas[x])
                .Max();

            return noise
                .MultiplyTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _sigmas = null;
            base.Dispose(disposing);
        }
    }
}
