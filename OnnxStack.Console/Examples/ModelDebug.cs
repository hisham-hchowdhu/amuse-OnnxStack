using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class ModelDebug : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public ModelDebug(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(ModelDebug));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 1;

        public string Name => "Model Debug";

        public string Description => "Model Debug";

        public async Task RunAsync()
        {
            // Create Pipeline
            // var pipeline = InstaFlowPipeline.CreatePipeline("D:\\Repositories\\Instaflow-onnx");
            // var pipeline = LatentConsistencyPipeline.CreatePipeline("D:\\Repositories\\LCM_Dreamshaper_v7-onnx");
            // var pipeline = LatentConsistencyXLPipeline.CreatePipeline("D:\\Repositories\\Latent-Consistency-xl-Olive-Onnx");
            // var pipeline = StableDiffusionPipeline.CreatePipeline("D:\\Repositories\\stable-diffusion-v1-5");
            // var pipeline = StableDiffusionXLPipeline.CreatePipeline("D:\\Repositories\\Hyper-SD-onnx");
            var pipeline = StableDiffusion3Pipeline.CreatePipeline("D:\\Repositories\\stable-diffusion-3-medium-diffusers");

            // Prompt
            var promptOptions = new PromptOptions
            {
                Prompt = "a photo of a cat drinking at a bar with a penguin"
            };

            // Scheduler
            var schedulerOptions = pipeline.DefaultSchedulerOptions with
            {
                Seed = 402437323,
                SchedulerType = SchedulerType.FlowMatchEulerDiscrete,
                GuidanceScale = 5f,
                InferenceSteps = 28
            };

            // Run pipeline
            var result = await pipeline.RunAsync(promptOptions, schedulerOptions, progressCallback: OutputHelpers.ProgressCallback);


            // Create Image from Tensor result
            var image = new OnnxImage(result);

            // Save Image File
            await image.SaveAsync(Path.Combine(_outputDirectory, $"{pipeline.GetType().Name}-{schedulerOptions.Seed}.png"));

            //Unload
            await pipeline.UnloadAsync();
        }
    }
}
