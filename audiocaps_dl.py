import hydra
from typing import Any, List, MutableMapping
import os
from yt_dlp import YoutubeDL
import csv
from loguru import logger
from omegaconf import DictConfig
from pathlib import Path
import glob, os
import subprocess
import librosa
import soundfile as sf
from multiprocessing import Pool, Process

def download_audio(yt_id, ydl_opts, mode, meta_list, settings, params):
    # Set logger
    logger_main = logger.bind(is_caption=False, indent=0)
    logger_sec = logger.bind(is_caption=False, indent=1)

    # Download audio
    with YoutubeDL(ydl_opts) as ydl:
        logger_sec.info(f'Try downloading {yt_id}')
        try:
            ydl.download([yt_id])
            logger_sec.info(f'Cut {yt_id}')
            cut_audio(mode, yt_id, meta_list, settings, params)
            logger_sec.info(f'Complete downloading {yt_id}.wav')
            
        except Exception as e:
            logger_sec.info(f'An error occured in downloading {yt_id}')
            logger_sec.info(f'Error message: {e}')

def cut_audio(mode, yt_id: str, meta_list: List[str], settings: MutableMapping[str, Any], params: MutableMapping[str, Any]) -> None: 
    # Get root dir
    root_dir = Path(settings['root_dir'])

    # Get target audio index in metadata
    meta_idx = [i[1] for i in meta_list].index(yt_id)

    # target audio path
    tgt_audio = str(root_dir.joinpath(settings['dirs']['audio_dir'] + '/' + mode + '/' + meta_list[meta_idx][1] + '_beforecutting.mp3'))

    # target audio is converted to wav_name
    wav_name = str(root_dir.joinpath(settings['dirs']['audio_dir'] + '/' + mode + '/' + meta_list[meta_idx][1] + '.wav'))

    # Cut audio and delete *.mp3 file
    subprocess.run([f"ffmpeg -ss {meta_list[meta_idx][2]} -i {tgt_audio} -t {params['ydl_opt']['seconds_to_cut']} {wav_name}"], shell=True) #meta_list[meta_idx][2]: start time
    subprocess.run([f'rm {tgt_audio}'], shell = True) #j[1]:youtube_id
    
def make_csv(mode: str, settings: MutableMapping[str, Any], params: MutableMapping[str, Any]) -> None: 
    # Get downloaded audio name
    downloaded_list = [os.path.split(i)[-1].split('.')[0] for i in glob.glob(str(root_dir.joinpath(settings['dirs']['audio_dir'] + '/' + mode + '/' + '*.wav')))]

    # Set header
    if mode == 'train':
        header = ['file_name','caption_1']

    elif mode == 'val' or mode == 'test':
        header = ['file_name','caption_1','caption_2','caption_3','caption_4','caption_5']

    caption_data = []
    caption_data.append(header)

    # Append audio name and captions
    for cnt, audio_name in enumerate(downloaded_list):
        index_num = [n for n, v in enumerate(meta_list) if audio_name in v]
        caption_data.append([audio_name])
        for j in index_num:
            caption_data[cnt+1] += [meta_list[j][3]]
    
    # Write to csv file
    with open(str(root_dir.joinpath(settings['dirs']['captions_csv_dir'] + '/AudioCaps_captions_' + mode + '.csv')), 'w') as f:
        writer = csv.writer(f)
        for i in caption_data:
            writer.writerow(i)
          
def audio_resampling(mode: str, settings: MutableMapping[str, Any], params: MutableMapping[str, Any]) -> None:
    # Get root_dir
    root_dir = Path(settings['root_dir'])

    # Set logger
    logger_main = logger.bind(is_caption=False, indent=0)
    logger_sec = logger.bind(is_caption=False, indent=1)

    # Get audio list
    audio_list = glob.glob(str(root_dir.joinpath(settings['dirs']['audio_dir'] + '/' + mode + '/*.wav')))
    
    # Resample audio
    for i in audio_list:
        logger_sec.info(f'Try resampling {i}')
        try:
            y, sr = librosa.core.load(i, sr=params['resample']['sr'], mono=params['resample']['mono'])
            sf.write(i, y, sr, subtype=params['resample']['subtype'])
            logger_sec.info(f'Complete resampling {i}')
        
        except Exception as e:
            logger_sec.info(f'Failed to resample {i} for {e}')
            logger_sec.info(f'mv {i}')
            subprocess.run([f"mv {i} {root_dir.joinpath(settings['dirs']['exceptional_audio_dir'])}"], shell = True) 

def read_csv(file_path: str, skip_head: bool) -> None:
    # Read csv file and return list
    with open(file_path, encoding='utf8', newline='') as f:
        tgt_list = []
        csvreader = csv.reader(f)
        if skip_head:
            header = next(csvreader)

        for row in csvreader:
            tgt_list.append(row)

    return tgt_list

def worker(id, mode, meta_list, settings, params):
    # Get root_dir
    root_dir = Path(settings['root_dir'])

    # Set logger
    logger_main = logger.bind(is_caption=False, indent=0)
    logger_sec = logger.bind(is_caption=False, indent=1)

    logger_main.info(f'Process {id}')

    # Get output directory
    output_dir = str(root_dir.joinpath(settings['dirs']['audio_dir'] + '/' + mode + '/%(id)s_beforecutting.%(ext)s'))
    
    # Set ydl options
    if settings['proxy']:
        ydl_opts = {'outtmpl': output_dir,
                    'formats': params['ydl_opt']['formats'], 
                    'postprocessors': [
                        {'key': params['ydl_opt']['postprocessors']['key'],
                        'preferredcodec': params['ydl_opt']['postprocessors']['preferredcodec'],
                        'preferredquality': params['ydl_opt']['postprocessors']['preferredquality']},
                        {'key': 'FFmpegMetadata'},
                    ],
                    'postprocessor_args': [
                        '-ar', params['ydl_opt']['sr'],
                    ],
                    'proxy': params['ydl_opt']['proxy_key'],
                    'ignoreerror': params['ydl_opt']['ignore_error'],
                    'cookiefile': settings['dirs']['cookie_path']
                    }
    else:
        ydl_opts = {'outtmpl': output_dir,
                    'formats':params['ydl_opt']['formats'], 
                    'postprocessors': [
                        {'key': params['ydl_opt']['postprocessors']['key'],
                        'preferredcodec': params['ydl_opt']['postprocessors']['preferredcodec'],
                        'preferredquality': params['ydl_opt']['postprocessors']['preferredquality']},
                        {'key': 'FFmpegMetadata'},
                    ],
                    'postprocessor_args': [
                        'ar', params['ydl_opt']['sr']
                    ],
                    'ignoreerror': params['ydl_opt']['ignore_error'],
                    'cookie': settings['dirs']['cookie_path']
                    }
    
    # Download audio
    download_audio(id, ydl_opts, mode, meta_list, settings, params)

# Set your xxx.yaml path here
@hydra.main(config_path='xxx', config_name = 'xxx')
def main(cfg:DictConfig):
    # Load settings
    workflow = cfg.workflow
    settings = cfg.settings
    params = cfg.params

    # Get root_dir
    root_dir = Path(settings['root_dir'])

    # Set logger
    logger_main = logger.bind(is_caption=False, indent=0)
    logger_sec = logger.bind(is_caption=False, indent=1)
    logger.add(
        str(root_dir.joinpath(settings['dirs']['log_path'])),
        )
    
    # Number of processes to be processed in parallel
    p = Pool(params['nprocess'])

    for mode in settings['mode']:
        if workflow['audio_download']:
            logger_main.info(f'Start downloading {mode} audio')

            # Get metadata
            meta_path = root_dir.joinpath(settings['dirs']['meta_dir'] + '/' + mode + '.csv')
            meta_list = read_csv(meta_path, skip_head=True)

            # Get youtube ids
            yt_id = [m[1] for m in meta_list]
            
            # If restart, exclude downloaded ids
            if settings['restart']:
                downloaded_list = [os.path.split(j)[-1].split('.')[0] for j in glob.glob(str(root_dir.joinpath(settings['dirs']['audio_dir'] + '/' + mode + '/' + '*.wav')))]
                yt_id = list(set(yt_id) - set(downloaded_list))

            # Setting up and execute parallel processing
            result = [p.apply_async(worker, (id, mode, meta_list, settings, params)) for id in yt_id]
            [f.get() for f in result]

        # Make csv file
        if workflow['make_captions_csv']:
            logger_main.info(f'Make {mode} captinos csv file')
            make_csv(mode, settings, params)
            logger_main.info(f'Complete {mode} captinos csv file')

        # Resample audio
        if workflow['resampling']:
            logger_main.info(f'Resample {mode} audio')
            audio_resampling(mode,settings,params)
            logger_main.info(f'Complete resampling {mode} audio')

if __name__ == '__main__':
    main()