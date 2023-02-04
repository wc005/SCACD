from Blocks import Decoder, Post, Prior, CausaConv, SCACD


def initSCACD(window, args):
    '''
    初始化模型
    :param window:
    :param args:
    :return:
    '''
    # 编码器 参数
    input_size = window
    encoder_input_dim = args.z_size
    encoder_hid_dim = args.hidden_size
    decoder_hidden_dim = encoder_hid_dim
    # 先验 Z 概率
    prior = Prior.autoencoder(input_size, args.hidden_size, args.z_size)
    # RNN 漂移网络
    ShiftModel = CausaConv.CausaConv(encoder_input_dim, encoder_hid_dim, args.n_layers, args.dropout)
    # 解码器
    decoder = Decoder.Decoder(args.z_size, decoder_hidden_dim, args.z_samples, args.s_samples, window)
    model = SCACD.SCACD(prior, ShiftModel, decoder)
    return model