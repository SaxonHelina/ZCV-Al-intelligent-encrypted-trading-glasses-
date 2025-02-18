from flask import Blueprint
from flask_restful import Api

from api.user.account import UserAccount, UserAccountAuth, UserAccountLog
from api.user.login import UserLogin, UserLoginLock
from api.user.role_list import Roles
from api.user.user_logs import UserOperationLogsList, UserOperationLogs, UserOperationLogsBatchDelete
from extensions import csrf_protect
from .user import UserList, User

user_blueprint = Blueprint('user', __name__, url_prefix='/')

user_api = Api(user_blueprint, prefix='/user', decorators=[csrf_protect.exempt],
               default_mediatype='application/json; charset=utf-8')

user_api.add_resource(UserLogin, '/login')
user_api.add_resource(UserList, '/list')
user_api.add_resource(User, '/', '/<int:user_id>', strict_slashes=False)
user_api.add_resource(Roles, '/roles')
user_api.add_resource(UserOperationLogsList, '/logs/list')
user_api.add_resource(UserOperationLogsBatchDelete, '/logs/batch_delete')
user_api.add_resource(UserOperationLogs, '/logs')
user_api.add_resource(UserLoginLock, '/lock')
user_api.add_resource(UserAccount, '/account/', '/account/<int:device_id>', strict_slashes=False)
user_api.add_resource(UserAccountAuth, '/account_auth/<int:account_id>')
user_api.add_resource(UserAccountLog, '/account_log/<int:account_id>')



