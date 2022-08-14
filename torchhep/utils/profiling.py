import json
import torchinfo


DEFAULT_MODEL_STATISTICS_ATTRS_BLACKLIST = ['summary_list', 'formatting']

def model_summary(model,
                  batch,
                  input_data,
                  device,
                  output_dir,
                  blacklist = DEFAULT_MODEL_STATISTICS_ATTRS_BLACKLIST,
):
    if isinstance(input_data, str):
        input_data = getattr(batch, input_data)
    elif isinstance(input_data, (list, tuple)):
        input_data = [getattr(batch, each) for each in input_data]
    else:
        raise TypeError(f'{type(input_data)=}')

    model_statistics = torchinfo.summary(
        model,
        input_data=input_data,
        device=device,
        mode='train')

    with open(output_dir / 'model_statistics.txt', 'w') as txt_file:
        txt_file.write(str(model_statistics))

    obj = vars(model_statistics)
    for key in blacklist:
        obj.pop(key)
    with open(output_dir / 'model_statistics.json', 'w') as json_file:
        json.dump(obj, json_file, indent=4)

    return model_statistics
