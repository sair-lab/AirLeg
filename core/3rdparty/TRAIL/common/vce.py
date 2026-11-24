"""VCE teacher generation; input maps and velocity exports remain immutable."""
from pathlib import Path

import yaml


def optimize_vce(backend, options, map_file, velocity, camera, calibration, directory, run, link,
                 gyro_bias=None):
    """gyro_bias: optional global-BA block {enabled, initial_csv, random_walk_sigma, prior_sigma}."""
    # numeric settings: alternating.vce.parameters (or the vce: block of a config_file); inputs set below
    parameters = (dict(options['parameters']) if options.get('parameters') is not None
                  else yaml.safe_load(Path(options['config_file']).read_text())['vce'])
    parameters.update(enabled=True, map_file=str(Path(map_file).resolve()),
                      velocity_csv=str(velocity.resolve()), camera_config=str(camera.resolve()),
                      T_C_B=calibration['extrinsic'], time_delay=calibration['time_delay'])
    # The enclosing run signature guards inputs/config. Reuse completed exports
    # after an interruption between publishing links and writing complete.json.
    for report in sorted(directory.glob('vce*/fusion/global_report.yaml'), reverse=True):
        fusion = report.parent
        if yaml.safe_load(report.read_text()).get('training_export_complete', False):
            return publish(fusion, directory, link)
    # Keep attempts immutable, including failed runs, when resuming an interrupted round.
    attempt = directory / 'vce'
    index = 0
    while attempt.exists():
        index += 1
        attempt = directory / f'vce_attempt_{index:03d}'
    attempt.mkdir()
    for mode, name in [('vce', 'windows'), ('vce-global', 'fusion')]:
        parameters['output_dir'] = str(attempt / name)
        if mode == 'vce-global':
            parameters.update(window_report=str(attempt / 'windows/report.yaml'),
                              export_training=True)
            if gyro_bias:
                parameters['gyro_bias'] = gyro_bias
        config = attempt / f'{name}.yaml'
        config.write_text(yaml.safe_dump({'vce': parameters}, sort_keys=False))
        run([backend, mode, config], attempt / f'{name}.log')
    return publish(attempt / 'fusion', directory, link)


def publish(fusion, directory, link):
    for name in ('AirSLAM_mapv1.bin', 'trajectory_v1.txt', 'calibration.yaml'):
        link(fusion / name, directory / name)
    link(fusion / 'leg_odom/keypoints', directory / 'leg_odom/keypoints')
    if (fusion / 'leg_odom/gyro_bias.csv').is_file():
        link(fusion / 'leg_odom/gyro_bias.csv', directory / 'gyro_bias.csv')
    return yaml.safe_load((fusion / 'global_report.yaml').read_text())['selected']
