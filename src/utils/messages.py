def rc_msg(data):
    return typed_msg('rc', data)

def cmd_msg(data):
    return typed_msg('cmd', data)

def typed_msg(type, data):
    return {'type': type, 'data': data}
