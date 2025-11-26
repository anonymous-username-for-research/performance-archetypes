import os
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..base_workload_generator import BaseWorkloadGenerator


class FFmpegWorkloadGenerator(BaseWorkloadGenerator):
    """Workload generator for FFmpeg video/audio processing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create temporary directory for test files
        self.temp_dir = os.path.join(self.output_dir, f"{self.program_name}_temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.test_files = {
            'video_testsrc': os.path.join(self.temp_dir, 'test_video.mp4'),
            'audio_sine': os.path.join(self.temp_dir, 'test_audio.mp3'),
            'video_color': os.path.join(self.temp_dir, 'test_color.mp4'),
            'image_test': os.path.join(self.temp_dir, 'test_image.png')
        }

        self.logger.info(f"FFmpeg workload generator initialized with temp dir: {self.temp_dir}")

    def _generate_test_files(self):
        """Generate test files for FFmpeg"""
        self.logger.info("Generating test input files...")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Parameters for test video
        test_video_params = {
            'duration': random.randint(2, 5),  # seconds
            'fps': random.randint(5, 15),
            'width': random.choice([240, 320, 480, 640]),
            'height': random.choice([180, 240, 360, 480])
        }

        # Generate test video with testsrc pattern
        if not os.path.exists(self.test_files['video_testsrc']):
            os.system(
                f"ffmpeg -f lavfi -i testsrc=duration={test_video_params['duration']}:"
                f"size={test_video_params['width']}x{test_video_params['height']}:"
                f"rate={test_video_params['fps']} "
                f"-c:v libx264 -preset ultrafast -crf 30 "
                f"-y {self.test_files['video_testsrc']} >/dev/null 2>&1"
            )

        # Generate test audio with sine wave
        if not os.path.exists(self.test_files['audio_sine']):
            os.system(
                f"ffmpeg -f lavfi -i sine=frequency=440:duration={test_video_params['duration']} "
                f"-y {self.test_files['audio_sine']} >/dev/null 2>&1"
            )

        # Generate solid color video
        if not os.path.exists(self.test_files['video_color']):
            os.system(
                f"ffmpeg -f lavfi -i color=c=blue:duration={test_video_params['duration']}:"
                f"s={test_video_params['width']}x{test_video_params['height']}:"
                f"r={test_video_params['fps']} "
                f"-c:v libx264 -preset ultrafast -crf 30 "
                f"-y {self.test_files['video_color']} >/dev/null 2>&1"
            )

        # Generate test image
        if not os.path.exists(self.test_files['image_test']):
            os.system(
                f"ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=1 "
                f"-frames:v 1 -y {self.test_files['image_test']} >/dev/null 2>&1"
            )

        self.logger.info("Test files generated successfully")

    def prepare_inputs(self, mode: str, from_db: bool = False,
                       db_query: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Prepare inputs for FFmpeg
        """
        operations = []

        # 1. Video Encoding Operations (different codecs and settings)
        presets = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium']
        crf_values = range(20, 35)
        codecs_video = []
        for preset in presets:
            for crf in crf_values:
                codecs_video.append(('libx264', 'mp4', {'preset': preset, 'crf': str(crf)}))
        
        for qscale in range(5, 15):
            codecs_video.append(('mpeg4', 'avi', {'qscale:v': str(qscale)}))

        for codec, ext, params in codecs_video:
            operations.append({
                'type': 'encode_video',
                'input': self.test_files['video_testsrc'],
                'codec': codec,
                'extension': ext,
                'params': params
            })

        # 2. Video Format Conversions
        format_conversions = [
            ('mp4', 'avi'),
            ('mp4', 'mkv'),
            ('mp4', 'webm'),
            ('avi', 'mp4'),
            ('mkv', 'mp4'),
            ('webm', 'mp4')
        ]

        for in_fmt, out_fmt in format_conversions:
            operations.append({
                'type': 'convert_format',
                'input': self.test_files['video_testsrc'],
                'output_format': out_fmt
            })

        # 3. Resolution Scaling Operations
        resolutions = []
        for width in range(160, 1280, 10):
            height = int(width * 3 / 4)
            resolutions.append((width, height))

        for width, height in resolutions:
            operations.append({
                'type': 'scale_video',
                'input': self.test_files['video_testsrc'],
                'width': width,
                'height': height
            })

        # 4. Frame Rate Changes
        frame_rates = list(range(10, 30, 1))

        for fps in frame_rates:
            operations.append({
                'type': 'change_fps',
                'input': self.test_files['video_testsrc'],
                'fps': fps
            })

        # 5. Video Filters
        filters = ['hflip', 'vflip', 'negate']
        for brightness in [round(x * 0.1, 1) for x in range(-5, 6)]:
            filters.append(f'eq=brightness={brightness}')
        for angle in range(0, 360, 10):
            filters.append(f'rotate={angle}*PI/180')
        for blur in range(1, 20):
            filters.append(f'boxblur={blur}:1')
        for x in range(0, 160, 20):
            for y in range(0, 120, 20):
                filters.append(f'crop=160:120:{x}:{y}')

        for filter_expr in filters:
            operations.append({
                'type': 'apply_filter',
                'input': self.test_files['video_testsrc'],
                'filter': filter_expr
            })

        # 6. Frame Extraction
        frame_extracts = list(range(1, 20))
        for num_frames in frame_extracts:
            operations.append({
                'type': 'extract_frames',
                'input': self.test_files['video_testsrc'],
                'num_frames': num_frames
            })

        # 7. Audio Operations
        audio_codecs = []
        for bitrate in ['64k', '96k', '128k', '192k', '256k', '320k']:
            audio_codecs.append(('libmp3lame', 'mp3', {'b:a': bitrate}))
            audio_codecs.append(('aac', 'm4a', {'b:a': bitrate}))
        for quality in range(0, 11):
            audio_codecs.append(('libvorbis', 'ogg', {'q:a': str(quality)}))

        for codec, ext, params in audio_codecs:
            operations.append({
                'type': 'encode_audio',
                'input': self.test_files['audio_sine'],
                'codec': codec,
                'extension': ext,
                'params': params
            })

        # 8. Audio Filters
        audio_filters = []
        for volume in [round(x * 0.1, 1) for x in range(1, 31)]:
            audio_filters.append(f'volume={volume}')
        for tempo in [round(x * 0.1, 1) for x in range(5, 21)]:
            audio_filters.append(f'atempo={tempo}')

        for filter_expr in audio_filters:
            operations.append({
                'type': 'audio_filter',
                'input': self.test_files['audio_sine'],
                'filter': filter_expr
            })

        # 9. Muxing Operations (combine audio and video)
        operations.append({
            'type': 'mux_av',
            'video_input': self.test_files['video_color'],
            'audio_input': self.test_files['audio_sine']
        })

        # 10. Trimming/Cutting
        trim_operations = []
        for start in [round(x * 0.1, 1) for x in range(0, 10)]:
            for duration in [round(x * 0.1, 1) for x in range(5, 15)]:
                trim_operations.append((start, duration))

        for start, duration in trim_operations:
            operations.append({
                'type': 'trim_video',
                'input': self.test_files['video_testsrc'],
                'start': start,
                'duration': duration
            })

        # 11. Image to Video
        for duration in range(1, 6):
            operations.append({
                'type': 'image_to_video',
                'input': self.test_files['image_test'],
                'duration': duration
            })

        # 12. Video to Images
        operations.append({
            'type': 'video_to_images',
            'input': self.test_files['video_testsrc']
        })

        # 13. Video concatenation
        for times in range(2, 6):
            operations.append({
                'type': 'concat_video',
                'input': self.test_files['video_testsrc'],
                'times': times
            })

        # 14. Bitrate changes
        bitrates = [f'{b}k' for b in range(50, 1000, 20)]

        for bitrate in bitrates:
            operations.append({
                'type': 'change_bitrate',
                'input': self.test_files['video_testsrc'],
                'bitrate': bitrate
            })

        operations = random.sample(operations, min(len(operations), self.num_data_points))

        self.logger.info(f"Prepared {len(operations)} diverse FFmpeg operations")

        return operations

    def prepare_commands(self, input_data: Any, build_type: str,
                         custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Prepare commands for FFmpeg
        """
        op_type = input_data['type']
        output_file = os.path.join(self.temp_dir, f"output_{random.randint(1000, 9999)}")

        command = ['ffmpeg_g', '-y']
        parameters = {'operation': op_type}

        self._generate_test_files()

        if op_type == 'encode_video':
            input_file = input_data['input']
            codec = input_data['codec']
            ext = input_data['extension']
            params = input_data['params']

            output_file = f"{output_file}.{ext}"
            command.extend(['-i', input_file, '-c:v', codec])

            for key, value in params.items():
                command.extend([f'-{key}', str(value)])

            command.append(output_file)
            parameters.update({'codec': codec, 'format': ext})

        elif op_type == 'convert_format':
            input_file = input_data['input']
            out_fmt = input_data['output_format']

            output_file = f"{output_file}.{out_fmt}"
            command.extend(['-i', input_file])

            if out_fmt == 'webm':
                command.extend(['-c:v', 'libvpx', '-crf', '30', '-b:v', '200k'])
            else:
                command.extend(['-c', 'copy'])

            command.append(output_file)
            parameters['output_format'] = out_fmt

        elif op_type == 'scale_video':
            input_file = input_data['input']
            width = input_data['width']
            height = input_data['height']

            output_file = f"{output_file}.mp4"
            command.extend([
                '-i', input_file,
                '-vf', f'scale={width}:{height}',
                '-c:v', 'libx264', '-preset', 'ultrafast',
                output_file
            ])
            parameters.update({'width': width, 'height': height})

        elif op_type == 'change_fps':
            input_file = input_data['input']
            fps = input_data['fps']

            output_file = f"{output_file}.mp4"
            command.extend([
                '-i', input_file,
                '-r', str(fps),
                '-c:v', 'libx264', '-preset', 'ultrafast',
                output_file
            ])
            parameters['fps'] = fps

        elif op_type == 'apply_filter':
            input_file = input_data['input']
            filter_expr = input_data['filter']

            output_file = f"{output_file}.mp4"
            command.extend([
                '-i', input_file,
                '-vf', filter_expr,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                output_file
            ])
            parameters['filter'] = filter_expr

        elif op_type == 'extract_frames':
            input_file = input_data['input']
            num_frames = input_data['num_frames']

            output_file = f"{output_file}_%03d.png"
            command.extend([
                '-i', input_file,
                '-vframes', str(num_frames),
                output_file
            ])
            parameters['num_frames'] = num_frames

        elif op_type == 'encode_audio':
            input_file = input_data['input']
            codec = input_data['codec']
            ext = input_data['extension']
            params = input_data['params']

            output_file = f"{output_file}.{ext}"
            command.extend(['-i', input_file, '-c:a', codec])

            for key, value in params.items():
                command.extend([f'-{key}', str(value)])

            command.append(output_file)
            parameters.update({'codec': codec, 'format': ext})

        elif op_type == 'audio_filter':
            input_file = input_data['input']
            filter_expr = input_data['filter']

            output_file = f"{output_file}.mp3"
            command.extend([
                '-i', input_file,
                '-af', filter_expr,
                output_file
            ])
            parameters['filter'] = filter_expr

        elif op_type == 'mux_av':
            video_input = input_data['video_input']
            audio_input = input_data['audio_input']

            output_file = f"{output_file}.mp4"
            command.extend([
                '-i', video_input,
                '-i', audio_input,
                '-c:v', 'copy', '-c:a', 'aac',
                '-shortest',
                output_file
            ])
            parameters['operation_detail'] = 'mux_audio_video'

        elif op_type == 'trim_video':
            input_file = input_data['input']
            start = input_data['start']
            duration = input_data['duration']

            output_file = f"{output_file}.mp4"
            command.extend([
                '-i', input_file,
                '-ss', str(start),
                '-t', str(duration),
                '-c', 'copy',
                output_file
            ])
            parameters.update({'start': start, 'duration': duration})

        elif op_type == 'image_to_video':
            input_file = input_data['input']
            duration = input_data['duration']

            output_file = f"{output_file}.mp4"
            command.extend([
                '-loop', '1',
                '-i', input_file,
                '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                output_file
            ])
            parameters['duration'] = duration

        elif op_type == 'video_to_images':
            input_file = input_data['input']

            output_file = f"{output_file}_%03d.jpg"
            command.extend([
                '-i', input_file,
                '-q:v', '5',
                output_file
            ])

        elif op_type == 'concat_video':
            input_file = input_data['input']
            times = input_data['times']

            concat_file = os.path.join(self.temp_dir, f'concat_{random.randint(1000, 9999)}.txt')
            with open(concat_file, 'w') as f:
                for _ in range(times):
                    f.write(f"file '{input_file}'\n")

            output_file = f"{output_file}.mp4"
            command.extend([
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output_file
            ])
            parameters['times'] = times

        elif op_type == 'change_bitrate':
            input_file = input_data['input']
            bitrate = input_data['bitrate']

            output_file = f"{output_file}.mp4"
            command.extend([
                '-i', input_file,
                '-b:v', bitrate,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                output_file
            ])
            parameters['bitrate'] = bitrate

        else:
            raise ValueError(f"Unknown operation type: {op_type}")

        self._current_output_file = output_file

        return command, parameters

    def _cleanup_after_input(self, input_data: Any) -> None:
        """Clean up temporary files after processing"""
        if hasattr(self, '_current_output_file'):
            output_file = self._current_output_file

            if '%' in output_file:
                import glob
                pattern = output_file.replace('%03d', '*')
                for file in glob.glob(pattern):
                    if os.path.exists(file):
                        os.remove(file)
            else:
                if os.path.exists(output_file):
                    os.remove(output_file)

            concat_files = [f for f in os.listdir(self.temp_dir) if f.startswith('concat_') and f.endswith('.txt')]
            for cf in concat_files:
                cf_path = os.path.join(self.temp_dir, cf)
                if os.path.exists(cf_path):
                    os.remove(cf_path)

        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Cleanup temp directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e}")
