from flask import current_app

from common.user import UserService as CommonUser


class UserService(CommonUser):
    def __init__(self, obj=None):
        super(UserService, self).__init__(obj)

    @staticmethod
    def get_basic_info(user, generate_token=True):
        base_info = {
            'user_id': user.id,
            'user_name': user.user_name,
            'user_role_id': user.roles[0].role_id,
            'user_role_text': user.roles[0].role.name,
        }

        if generate_token:
            base_info['token'] = user.generate_auth_token(expiration=current_app.config['TOKEN_EXPIRATION_TIME'])

        return base_info
