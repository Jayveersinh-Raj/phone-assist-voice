"""
Deepgram STT provider implementation.
"""
import os
import asyncio
from typing import Optional, Dict, Any
import numpy as np
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import tempfile
import soundfile as sf
from .base import STTProvider


class DeepgramSTT(STTProvider):
    """Deepgram Speech-to-Text provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Deepgram STT provider.
        
        Args:
            config: Configuration dictionary with 'api_key' and optional settings
        """
        super().__init__(config)
        
        # Get API key from config or environment
        api_key = self.config.get('api_key') or os.getenv('DEEPGRAM_API_KEY')
        if not api_key:
            raise ValueError("Deepgram API key is required. Set DEEPGRAM_API_KEY environment variable or pass in config.")
        
        self.client = DeepgramClient(api_key)
        self.language = self.config.get('language', 'en')
        self.model = self.config.get('model', 'nova-2')
        self.smart_format = self.config.get('smart_format', True)
        self.punctuate = self.config.get('punctuate', True)
        
        # Supported languages (subset of Deepgram's supported languages)
        self.supported_languages = [
            'en', 'hi', 'gu', 'bn', 'ta', 'te', 'ml', 'kn', 'mr', 'pa', 'or', 'as',
            'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'tr', 'pl',
            'nl', 'sv', 'da', 'no', 'fi', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl',
            'et', 'lv', 'lt', 'mt', 'cy', 'ga', 'is', 'mk', 'sq', 'sr', 'uk', 'be',
            'ka', 'hy', 'az', 'kk', 'ky', 'uz', 'mn', 'ne', 'si', 'my', 'km', 'lo',
            'th', 'vi', 'id', 'ms', 'tl', 'sw', 'am', 'ha', 'ig', 'yo', 'zu', 'xh',
            'af', 'sq', 'eu', 'ca', 'gl', 'is', 'ga', 'cy', 'mt', 'mk', 'sq', 'sr',
            'uk', 'be', 'ka', 'hy', 'az', 'kk', 'ky', 'uz', 'mn', 'ne', 'si', 'my',
            'km', 'lo', 'th', 'vi', 'id', 'ms', 'tl', 'sw', 'am', 'ha', 'ig', 'yo',
            'zu', 'xh', 'af'
        ]
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using Deepgram.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        try:
            # Convert numpy array to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert to float32 and normalize
                audio_float = audio_data.astype(np.float32) / 32768.0
                sf.write(temp_file.name, audio_float, sample_rate)
                
                # Read the file for Deepgram
                with open(temp_file.name, 'rb') as audio_file:
                    buffer_data = audio_file.read()
                
                # Clean up temp file
                os.unlink(temp_file.name)
            
            # Configure Deepgram options
            options = PrerecordedOptions(
                model=self.model,
                language=self.language,
                smart_format=self.smart_format,
                punctuate=self.punctuate,
                diarize=False,
                multichannel=False
            )
            
            # Create file source
            payload: FileSource = {
                "buffer": buffer_data,
            }
            
            # Make API call
            response = self.client.listen.prerecorded.v("1").transcribe_file(
                payload, options
            )
            
            # Extract transcript
            if response.results and response.results.channels:
                transcript = response.results.channels[0].alternatives[0].transcript
                return transcript.strip()
            
            return ""
            
        except Exception as e:
            print(f"Deepgram transcription error: {e}")
            return ""
    
    def transcribe_streaming(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a streaming audio chunk using Deepgram.
        For streaming, we'll use the same method as regular transcription.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text for this chunk
        """
        return self.transcribe(audio_chunk, sample_rate)
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def set_language(self, language: str) -> None:
        """
        Set the language for transcription.
        
        Args:
            language: Language code
        """
        if language in self.supported_languages:
            self.language = language
            self.config['language'] = language
        else:
            raise ValueError(f"Language '{language}' not supported. Supported languages: {self.supported_languages}")
    
    def update_model(self, model: str) -> None:
        """
        Update the Deepgram model.
        
        Args:
            model: Model name (e.g., 'nova-2', 'base', 'enhanced')
        """
        self.model = model
        self.config['model'] = model
