from common.base import RestfulBase
from common.base import success_response
from common.base import parse_page
from api.security import user_login_required as login_required
from common.base.api import get_per_page
from models.user import Role
from models.user.role_config import ROLES


class RoleList(RestfulBase):


    @login_required
    def get(self):
        roles = Role.query.filter(Role.active == True)
        items = {}
        for item in roles:
            items[item.id] = item.name
        return success_response(data=items)


class Roles(RestfulBase):

    # @login_required
    def get(self):
        args = parse_page()
        page = args['page']
        per_page = get_per_page(args)

        pagination = Role.query.filter(Role.active == True, Role.id != ROLES['ADMINISTRATOR']).paginate(page, per_page)
        items = []
        for item in pagination.items:
            items.append({
                'id': item.id,
                'name': item.name,
                'parent_id': item.parent_id,
            })

        res = {
            'total': pagination.total,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'items': items
        }
        return success_response(data=res)
