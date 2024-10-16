﻿using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public class StableDiffusionXLPipeline : StableDiffusionPipeline
    {
        protected TokenizerModel _tokenizer2;
        protected TextEncoderModel _textEncoder2;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLPipeline"/> class.
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
        public StableDiffusionXLPipeline(PipelineOptions pipelineOptions, TokenizerModel tokenizer, TokenizerModel tokenizer2, TextEncoderModel textEncoder, TextEncoderModel textEncoder2, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, UNetConditionModel controlNet, List<DiffuserType> diffusers, List<SchedulerType> schedulers = default, SchedulerOptions defaultSchedulerOptions = default, ILogger logger = default)
            : base(pipelineOptions, tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder, controlNet, diffusers, schedulers, defaultSchedulerOptions, logger)
        {
            _tokenizer2 = tokenizer2;
            _textEncoder2 = textEncoder2;
            _supportedSchedulers = schedulers ?? new List<SchedulerType>
            {
                SchedulerType.Euler,
                SchedulerType.EulerAncestral,
                SchedulerType.DDPM,
                SchedulerType.KDPM2,
                SchedulerType.DDIM
            };
            if (_unet.ModelType == ModelType.Turbo)
            {
                _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
                {
                    Width = 512,
                    Height = 512,
                    InferenceSteps = 2,
                    GuidanceScale = 0f,
                    SchedulerType = SchedulerType.EulerAncestral
                };
            }
            else
            {
                _defaultSchedulerOptions = defaultSchedulerOptions ?? new SchedulerOptions
                {
                    Width = 1024,
                    Height = 1024,
                    InferenceSteps = 20,
                    GuidanceScale = 5f,
                    SchedulerType = SchedulerType.EulerAncestral
                };
            }
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.StableDiffusionXL;


        /// <summary>
        /// Gets the tokenizer2.
        /// </summary>
        public TokenizerModel Tokenizer2 => _tokenizer2;


        /// <summary>
        /// Gets the text encoder2.
        /// </summary>
        public TextEncoderModel TextEncoder2 => _textEncoder2;


        /// <summary>
        /// Loads the pipeline
        /// </summary>
        public override Task LoadAsync(UnetModeType unetMode = UnetModeType.Default)
        {
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
                return base.LoadAsync(unetMode);

            // Preload all models into VRAM
            return Task.WhenAll
            (
                _tokenizer2.LoadAsync(),
                _textEncoder2.LoadAsync(),
                base.LoadAsync(unetMode)
            );
        }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns></returns>
        public override Task UnloadAsync()
        {
            _tokenizer2?.Dispose();
            _textEncoder2?.Dispose();
            return base.UnloadAsync();
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
                DiffuserType.ImageToImage => new ImageDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ImageInpaintLegacy => new InpaintLegacyDiffuser(_unet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ControlNet => new ControlNetDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
                DiffuserType.ControlNetImage => new ControlNetImageDiffuser(controlNetModel, _controlNetUnet, _vaeDecoder, _vaeEncoder, _pipelineOptions.MemoryMode, _logger),
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
            return _unet.ModelType switch
            {
                ModelType.Refiner => await CreateEmbedsTwoAsync(promptOptions, isGuidanceEnabled),
                _ => await CreateEmbedsBothAsync(promptOptions, isGuidanceEnabled),
            };
        }


        /// <summary>
        /// Creates the embeds using Tokenizer2 and TextEncoder2
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> [is guidance enabled].</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsTwoAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var promptTokens = await DecodeTextAsLongAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodeTextAsLongAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GenerateEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GenerateEmbedsAsync(negativePromptTokens, maxPromptTokenCount);

            // Unload if required
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
            {
                await _tokenizer2.UnloadAsync();
                await _textEncoder2.UnloadAsync();
            }

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(
                    negativePromptEmbeddings.PromptEmbeds.Concatenate(promptEmbeddings.PromptEmbeds),
                    negativePromptEmbeddings.PooledPromptEmbeds.Concatenate(promptEmbeddings.PooledPromptEmbeds));

            return new PromptEmbeddingsResult(promptEmbeddings.PromptEmbeds, promptEmbeddings.PooledPromptEmbeds);
        }


        /// <summary>
        /// Creates the embeds using Tokenizer, Tokenizer2, TextEncoder and TextEncoder2
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="isGuidanceEnabled">if set to <c>true</c> is guidance enabled.</param>
        /// <returns></returns>
        private async Task<PromptEmbeddingsResult> CreateEmbedsBothAsync(PromptOptions promptOptions, bool isGuidanceEnabled)
        {
            // Tokenize Prompt and NegativePrompt
            var promptTokens = await DecodePromptTextAsync(promptOptions.Prompt);
            var negativePromptTokens = await DecodePromptTextAsync(promptOptions.NegativePrompt);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Generate embeds for tokens
            var promptEmbeddings = await GeneratePromptEmbedsAsync(promptTokens, maxPromptTokenCount);
            var negativePromptEmbeddings = await GeneratePromptEmbedsAsync(negativePromptTokens, maxPromptTokenCount);

            /// Tokenize Prompt and NegativePrompt with Tokenizer2
            var dualPromptTokens = await DecodeTextAsLongAsync(promptOptions.Prompt);
            var dualNegativePromptTokens = await DecodeTextAsLongAsync(promptOptions.NegativePrompt);

            // Generate embeds for tokens
            var dualPromptEmbeddings = await GenerateEmbedsAsync(dualPromptTokens, maxPromptTokenCount);
            var dualNegativePromptEmbeddings = await GenerateEmbedsAsync(dualNegativePromptTokens, maxPromptTokenCount);

            var dualPrompt = promptEmbeddings.PromptEmbeds.Concatenate(dualPromptEmbeddings.PromptEmbeds, 2);
            var dualNegativePrompt = negativePromptEmbeddings.PromptEmbeds.Concatenate(dualNegativePromptEmbeddings.PromptEmbeds, 2);
            var pooledPromptEmbeds = dualPromptEmbeddings.PooledPromptEmbeds;
            var pooledNegativePromptEmbeds = dualNegativePromptEmbeddings.PooledPromptEmbeds;

            // Unload if required
            if (_pipelineOptions.MemoryMode == MemoryModeType.Minimum)
            {
                await _tokenizer2.UnloadAsync();
                await _textEncoder2.UnloadAsync();
            }

            if (isGuidanceEnabled)
                return new PromptEmbeddingsResult(dualNegativePrompt.Concatenate(dualPrompt), pooledNegativePromptEmbeds.Concatenate(pooledPromptEmbeds));

            return new PromptEmbeddingsResult(dualPrompt, pooledPromptEmbeds);
        }


        /// <summary>
        /// Decodes the text as tokens
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <returns></returns>
        protected async Task<TokenizerResult> DecodeTextAsLongAsync(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return new TokenizerResult(Array.Empty<long>(), Array.Empty<long>());

            var metadata = await _tokenizer2.GetMetadataAsync();
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            var timestamp = _logger.LogBegin();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(inputTensor);
                inferenceParameters.AddOutputBuffer();
                inferenceParameters.AddOutputBuffer();
                using (var results = _tokenizer.RunInference(inferenceParameters))
                {
                    _logger?.LogEnd("_tokenizer ", timestamp);
                    return new TokenizerResult(results[0].ToArray<long>(), results[1].ToArray<long>());
                }
            }
        }


        /// <summary>
        /// Encodes the tokens.
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        private async Task<EncoderResult> EncodeTokensAsync(TokenizerResult tokenizedInput)
        {
            var metadata = await _textEncoder2.GetMetadataAsync();
            var inputTensor = new DenseTensor<long>(tokenizedInput.InputIds, new[] { 1, tokenizedInput.InputIds.Length });

            var timestamp = _logger.LogBegin();

            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                int hiddenStateIndex = metadata.Outputs.Count - 2;
                inferenceParameters.AddInputTensor(inputTensor);

                // text_embeds + hidden_states.31 ("31" because SDXL always indexes from the penultimate layer.)
                inferenceParameters.AddOutputBuffer(new[] { 1, _tokenizer2.TokenizerLength });
                inferenceParameters.AddOutputBuffer(hiddenStateIndex, new[] { 1, tokenizedInput.InputIds.Length, _tokenizer2.TokenizerLength });

                var results = await _textEncoder2.RunInferenceAsync(inferenceParameters);
                using (var promptEmbeds = results.Last())
                using (var promptEmbedsPooled = results.First())
                {
                    _logger?.LogEnd("_textEncoder2 ", timestamp);
                    return new EncoderResult(promptEmbeds.ToDenseTensor(), promptEmbedsPooled.ToDenseTensor());
                }
            }
        }


        /// <summary>
        /// Generates the embeds.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <returns></returns>
        protected async Task<PromptEmbeddingsResult> GenerateEmbedsAsync(TokenizerResult inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.InputIds.Length < minimumLength)
            {
                inputTokens.InputIds = PadWithBlankTokens(inputTokens.InputIds, minimumLength, _tokenizer.PadTokenId).ToArray();
                inputTokens.AttentionMask = PadWithBlankTokens(inputTokens.AttentionMask, minimumLength, 1).ToArray();
            }

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate1
            var tokenBatches = new List<long[]>();
            var attentionBatches = new List<long[]>();
            foreach (var tokenBatch in inputTokens.InputIds.Chunk(_tokenizer.TokenizerLimit))
                tokenBatches.Add(PadWithBlankTokens(tokenBatch, _tokenizer.TokenizerLimit, _tokenizer.PadTokenId).ToArray());
            foreach (var attentionBatch in inputTokens.AttentionMask.Chunk(_tokenizer.TokenizerLimit))
                attentionBatches.Add(PadWithBlankTokens(attentionBatch, _tokenizer.TokenizerLimit, 1).ToArray());

            var promptEmbeddings = new List<float>();
            var pooledPromptEmbeddings = new List<float>();
            for (int i = 0; i < tokenBatches.Count; i++)
            {
                var result = await EncodeTokensAsync(new TokenizerResult(tokenBatches[i], attentionBatches[i]));
                promptEmbeddings.AddRange(result.PromptEmbeds);
                pooledPromptEmbeddings.AddRange(result.PooledPromptEmbeds);
            }

            var promptTensor = new DenseTensor<float>(promptEmbeddings.ToArray(), new[] { 1, promptEmbeddings.Count / _tokenizer2.TokenizerLength, _tokenizer2.TokenizerLength });
            var pooledTensor = new DenseTensor<float>(pooledPromptEmbeddings.Take(_tokenizer2.TokenizerLength).ToArray(), new[] { 1, _tokenizer2.TokenizerLength });
            return new PromptEmbeddingsResult(promptTensor, pooledTensor);
        }


        /// <summary>
        /// Creates the pipeline from a ModelSet configuration.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static new StableDiffusionXLPipeline CreatePipeline(StableDiffusionModelSet modelSet, ILogger logger = default)
        {
            var config = modelSet with { };
            var unet = new UNetConditionModel(config.UnetConfig.ApplyDefaults(config));
            var tokenizer = new TokenizerModel(config.TokenizerConfig.ApplyDefaults(config));
            var tokenizer2 = new TokenizerModel(config.Tokenizer2Config.ApplyDefaults(config));
            var textEncoder = new TextEncoderModel(config.TextEncoderConfig.ApplyDefaults(config));
            var textEncoder2 = new TextEncoderModel(config.TextEncoder2Config.ApplyDefaults(config));
            var vaeDecoder = new AutoEncoderModel(config.VaeDecoderConfig.ApplyDefaults(config));
            var vaeEncoder = new AutoEncoderModel(config.VaeEncoderConfig.ApplyDefaults(config));
            var controlnet = default(UNetConditionModel);
            if (config.ControlNetUnetConfig is not null)
                controlnet = new UNetConditionModel(config.ControlNetUnetConfig.ApplyDefaults(config));

            var pipelineOptions = new PipelineOptions(config.Name, config.MemoryMode);
            return new StableDiffusionXLPipeline(pipelineOptions, tokenizer, tokenizer2, textEncoder, textEncoder2, unet, vaeDecoder, vaeEncoder, controlnet, config.Diffusers, config.Schedulers, config.SchedulerOptions, logger);
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
        public static new StableDiffusionXLPipeline CreatePipeline(string modelFolder, ModelType modelType = ModelType.Base, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, MemoryModeType memoryMode = MemoryModeType.Maximum, ILogger logger = default)
        {
            return CreatePipeline(ModelFactory.CreateModelSet(modelFolder, DiffuserPipelineType.StableDiffusionXL, modelType, deviceId, executionProvider, memoryMode), logger);
        }
    }
}
