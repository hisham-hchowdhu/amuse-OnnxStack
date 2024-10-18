using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusion3;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class StableDiffusion3Pipeline : StableDiffusionXLPipeline
    {
        private readonly TokenizerModel _tokenizer3;
        private readonly TextEncoderModel _textEncoder3;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Pipeline"/> class.
        /// </summary>
        /// <param name="pipelineOptions">The pipeline options</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusion3Pipeline(PipelineOptions pipelineOptions, TokenizerModel tokenizer, TokenizerModel tokenizer2, TokenizerModel tokenizer3, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, TextEncoderModel textEncoder3, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(pipelineOptions, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _tokenizer3 = tokenizer3;
            _textEncoder3 = textEncoder3;
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.FlowMatchEulerDiscrete
            };
            _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
            {
                Width = 1024,
                Height = 1024,
                InferenceSteps = 28,
                GuidanceScale = 7f,
                SchedulerType = SchedulerType.FlowMatchEulerDiscrete
            };
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableDiffusion3;


        /// <summary>
        /// Loads the pipeline
        /// </summary>
        public override Task LoadAsync(UnetModeType unetMode = UnetModeType.Default)
        {
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
                return base.LoadAsync(unetMode);

            var tokenizer3Task = Task.CompletedTask;
            var textEncoder3Task = Task.CompletedTask;
            if (_tokenizer3 is not null)
                tokenizer3Task = _tokenizer3.LoadAsync();
            if (_textEncoder3 is not null)
                textEncoder3Task = _textEncoder3.LoadAsync();

            return Task.WhenAll
            (
                tokenizer3Task,
                textEncoder3Task,
                base.LoadAsync(unetMode)
            );
        }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public override Task UnloadAsync()
        {
            _tokenizer3?.Dispose();
            _textEncoder3?.Dispose();
            return base.UnloadAsync();
        }


        /// <summary>
        /// Runs the pipeline.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override Task<DenseTensor<float>> RunAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return base.RunAsync(promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Runs the pipeline batch.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override IAsyncEnumerable<BatchResult> RunBatchAsync(BatchOptions batchOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions = default, ControlNetModel controlNet = default, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return base.RunBatchAsync(batchOptions, promptOptions, schedulerOptions, controlNet, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Creates the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <param name="controlNetModel">The control net model.</param>
        /// <returns></returns>
        protected override IDiffuser CreateDiffuser(DiffuserType diffuserType, ControlNetModel controlNetModel)
        {
            return diffuserType switch
            {
                DiffuserType.TextToImage => new TextDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                _ => throw new NotImplementedException()
            };
        }

        /// <summary>
        /// Creates the prompt embeds.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> [is guidance enabled].</param>
        /// <returns></returns>
        protected override async Task<PromptEmbeddingsResult> CreatePromptEmbedsAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodePromptTextAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodePromptTextAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount);

            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var prompt2Tokens = await DecodeTextAsLongAsync(promptOptions.Prompt);
            var negativePrompt2Tokens = await DecodeTextAsLongAsync(promptOptions.NegativePrompt);

            // Generate embeds for tokens with TextEncoder2
            var prompt2Embeddings = await GenerateEmbedsAsync(prompt2Tokens, maxPromptTokenCount);
            var negativePrompt2Embeddings = await GenerateEmbedsAsync(negativePrompt2Tokens, maxPromptTokenCount);

            /// Tokenize Prompt and NegativePrompt with Tokenizer3
            var prompt3Tokens = await TokenizePrompt3Async(promptOptions.Prompt);
            var negativePrompt3Tokens = await TokenizePrompt3Async(promptOptions.NegativePrompt);

            // Generate embeds for tokens with TextEncoder3
            var prompt3Embeddings = await GeneratePrompt3EmbedsAsync(prompt3Tokens, maxPromptTokenCount);
            var negativePrompt3Embeddings = await GeneratePrompt3EmbedsAsync(negativePrompt3Tokens, maxPromptTokenCount);


            // Positive Prompt
            var prompt_embed = promptEmbeddings.PromptEmbeds;

            // We batch CLIP greater than 77 not truncate so the pool embeds wont work that way, so only use first set
            var pooled_prompt_embed = promptEmbeddings.PooledPromptEmbeds
                .ReshapeTensor([promptEmbeddings.PooledPromptEmbeds.Dimensions[^2], promptEmbeddings.PooledPromptEmbeds.Dimensions[^1]])
                .FirstBatch();

            var prompt_2_embed = prompt2Embeddings.PromptEmbeds;
            var pooled_prompt_2_embed = prompt2Embeddings.PooledPromptEmbeds.FirstBatch();

            var clip_prompt_embeds = prompt_embed.Concatenate(prompt_2_embed, 2);
            clip_prompt_embeds = clip_prompt_embeds.PadEnd(prompt3Embeddings.Dimensions[^1] - clip_prompt_embeds.Dimensions[^1]);
            var prompt_embeds = clip_prompt_embeds.Concatenate(prompt3Embeddings, 1);

            pooled_prompt_2_embed = pooled_prompt_2_embed.Repeat(pooled_prompt_embed.Dimensions[0]);
            var pooled_prompt_embeds = pooled_prompt_embed.Concatenate(pooled_prompt_2_embed, 1);


            // Negative Prompt
            var negative_prompt_embed = negativePromptEmbeddings.PromptEmbeds;

            // We batch CLIP greater than 77 not truncate so the pool embeds wont work that way, so only use first set
            var negative_pooled_prompt_embed = negativePromptEmbeddings.PooledPromptEmbeds
                .ReshapeTensor([negativePromptEmbeddings.PooledPromptEmbeds.Dimensions[^2], negativePromptEmbeddings.PooledPromptEmbeds.Dimensions[^1]])
                .FirstBatch();

            var negative_prompt_2_embed = negativePrompt2Embeddings.PromptEmbeds;
            var negative_pooled_prompt_2_embed = negativePrompt2Embeddings.PooledPromptEmbeds.FirstBatch();

            var negative_clip_prompt_embeds = negative_prompt_embed.Concatenate(negative_prompt_2_embed, 2);
            negative_clip_prompt_embeds = negative_clip_prompt_embeds.PadEnd(negativePrompt3Embeddings.Dimensions[^1] - negative_clip_prompt_embeds.Dimensions[^1]);
            var negative_prompt_embeds = negative_clip_prompt_embeds.Concatenate(negativePrompt3Embeddings, 1);

            negative_pooled_prompt_2_embed = negative_pooled_prompt_2_embed.Repeat(negative_pooled_prompt_embed.Dimensions[0]);
            var negative_pooled_prompt_embeds = negative_pooled_prompt_embed.Concatenate(negative_pooled_prompt_2_embed, 1);


            // Unload if required
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
            {
                await _tokenizer.UnloadAsync();
                await _textEncoder.UnloadAsync();
                await _tokenizer2.UnloadAsync();
                await _textEncoder2.UnloadAsync();
                if (_tokenizer3 is not null)
                    await _tokenizer3.UnloadAsync();
                if (_textEncoder3 is not null)
                    await _textEncoder3.UnloadAsync();
            }

            // If guidance is enabled, contact negative and positive prompts
            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(
                    negative_prompt_embeds.Concatenate(prompt_embeds),
                    negative_pooled_prompt_embeds.Concatenate(pooled_prompt_embeds));

            return new PromptEmbeddingsResult(prompt_embeds, pooled_prompt_embeds);
        }


        /// <summary>
        /// Generates the prompt3 embeds asynchronous.
        /// </summary>
        /// <param name="prompt3Tokens">The prompt3 tokens.</param>
        /// <param name="maxPromptTokenCount">The maximum prompt token count.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> GeneratePrompt3EmbedsAsync(TokenizerResult prompt3Tokens, int maxPromptTokenCount)
        {
            if (prompt3Tokens is null || prompt3Tokens.InputIds.Length == 0)
                return new DenseTensor<float>([1, Math.Max(77, maxPromptTokenCount), 4096]);

            var metadata = await _textEncoder3.GetMetadataAsync();
            var inputTensor = new DenseTensor<int>(prompt3Tokens.InputIds.ToInt(), [1, prompt3Tokens.InputIds.Length]);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();

                using (var results = _textEncoder3.RunInference(inferenceParameters))
                {
                    return results[0].ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Decodes the text as tokens
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        private async Task<TokenizerResult> TokenizePrompt3Async(string inputText)
        {
            if (_tokenizer3 is null)
                return null;

            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult([], []);

            var metadata = await _tokenizer3.GetMetadataAsync();
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, [1]);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();
                using (var results = _tokenizer3.RunInference(inferenceParameters))
                {
                    return new TokenizerResult(Array.ConvertAll(results[0].ToArray<int>(), Convert.ToInt64), []);
                }
            }
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableDiffusion3Pipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new TokenizerModel(config.TokenizerConfig.ApplyDefaults(config));
            var tokenizer2 = new TokenizerModel(config.Tokenizer2Config.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var textEncoder2 = new TextEncoderModel(config.TextEncoder2Config.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));

            var tokenizer3 = default(TokenizerModel);
            if (config.Tokenizer3Config is not null)
            {
                // SentencePiece tokenizer only supports CPU
                config.Tokenizer3Config.ExecutionProvider = ExecutionProvider.Cpu;
                tokenizer3 = new TokenizerModel(config.Tokenizer3Config.ApplyDefaults(config));
            }

            var textEncoder3 = default(TextEncoderModel);
            if (config.TextEncoder3Config is not null)
                textEncoder3 = new TextEncoderModel(config.TextEncoder3Config.ApplyDefaults(config));

            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            var pipelineOptions = new PipelineOptions(config.Name, config.MemoryMode);
            return new StableDiffusion3Pipeline(pipelineOptions, tokenizer, tokenizer2, tokenizer3, textEncoder, textEncoder2, textEncoder3, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
        }


        /// <summary>
        /// Creates the pipeline from a folder structure.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableDiffusion3Pipeline CreatePipeline(string modelFolder, ModelType modelType = ModelType.Base, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, MemoryModeType memoryMode = MemoryModeType.Maximum, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, DiffuserPipelineType.StableDiffusion3, modelType, deviceId, executionProvider, memoryMode), logger);
        }
    }
}
