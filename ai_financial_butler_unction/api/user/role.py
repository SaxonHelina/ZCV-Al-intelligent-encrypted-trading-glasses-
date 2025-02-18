from api.security import user_login_required as login_required
from common.base import RestfulBase
from common.base import raise_404_response
from common.base import success_response
from models.user import Role as RoleModel


class Roles(RestfulBase):

    @login_required
    def get(self, role_id):
        role = RoleModel.query.filter(RoleModel.id == role_id, RoleModel.active == True).first()
        if not role:
            raise_404_response(message='123')

        data = {
            'id': role.id,
            'name': role.name,
            'parent_id': role.parent_id,
        }
        return success_response(data=data)
