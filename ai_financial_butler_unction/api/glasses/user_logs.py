from flask import current_app, g
from flask_restful import reqparse
from sqlalchemy import desc

from api.security import role_required
from common.base import RestfulBase, parse_page, raise_400_response, raise_404_response
from common.base import success_response
from common.base.api import get_per_page
from common.user.service import UserOperationService
from common.user.user import get_user_ip
from common.utils import validator
from common.utils.dt import loop_delete_log_table
from extensions import db
from models import SystemLog
from models.user import User
from models.user.role_config import ROLES
from models.user.user_operation import UserOperation
from ..security import user_login_required as login_required


class UserOperationLogsList(RestfulBase):

    @login_required
    @role_required(ROLES['ADMINISTRATOR'], ROLES['COMPTROLLER'])
    def get(self):
        sorted_fields = {
            'user_id': UserOperation.id,
        }

        args = parse_page(sorted_fields=sorted_fields, default_sort_field=UserOperation.id)
        page = args['page']
        per_page = get_per_page(args)

        rep = reqparse.RequestParser()
        rep.add_argument('user_name', type=validator.validate_max50)
        rep.add_argument('time_from', type=validator.validate_datetime)
        rep.add_argument('time_to', type=validator.validate_datetime)
        rep.add_argument('export', type=bool)
        rep.add_argument('device_ip', type=str)
        rep.add_argument('user_ip', type=str)
        rep.add_argument('detail', type=str)
        query_args = rep.parse_args()

        query = UserOperation.query.order_by(desc(UserOperation.created_at), desc(UserOperation.id))

        if query_args.user_name:
            query = query.join(User, User.id == UserOperation.user_id) \
                .filter(User.user_name.ilike('%{}%'.format(query_args.user_name).replace('\b', '').replace('_', '\_')))

        if query_args.time_from:
            query = query.filter(UserOperation.created_at >= query_args.time_from)
        if query_args.time_to:
            query = query.filter(UserOperation.created_at <= query_args.time_to)

        if query_args.device_ip:
            query = query.filter \
                    (UserOperation.device_ip.ilike(
                    '%{}%'.format(query_args.device_ip).replace('\b', '').replace('_', '\_')))

        if query_args.user_ip:
            query = query.filter \
                (UserOperation.user_ip.ilike('%{}%'.format(query_args.user_ip).replace('\b', '').replace('_', '\_')))

        if query_args.detail:
            query = query.filter \
                (UserOperation.memo.ilike('%{}%'.format(query_args.detail).replace('\b', '').replace('_', '\_')))

        pagination = query.paginate(page, per_page, error_out=current_app.config['DEBUG'])
        items = []
        for item in pagination.items:
            operation_info = UserOperationService.render(item)
            items.append(operation_info)

        res = {
            'total': pagination.total,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'items': items
        }
        if query_args.export:
            user_name_text = ''
            user_ip = ''
            time_text = ''
            device_ip = ''
            detail_text = ''
            if query_args.user_name:
                user_name_text = ',1: %s' % query_args.user_name
            if query_args.user_ip:
                user_ip = ',1: %s' % query_args.user_ip
            if query_args.time_from and query_args.time_to:
                time_text = ',1: %s -- %s' % (query_args.time_from.strftime("%Y-%m-%d  %H:%M:%S"),
                                               query_args.time_to.strftime("%Y-%m-%d  %H:%M:%S"))
            if query_args.device_ip:
                device_ip = ',1: %s' % query_args.device_ip
            if query_args.detail:
                detail_text = ',1: %s' % query_args.detail
            loop_delete_log_table(SystemLog, True)
            SystemLog.create(
                memo='1' % (g.current_user.user_name, get_user_ip())
                     + user_name_text + user_ip + time_text + device_ip + detail_text,
            )
        return success_response(data=res)


class UserOperationLogsBatchDelete(RestfulBase):
    @login_required
    @role_required(ROLES['ADMINISTRATOR'], ROLES['COMPTROLLER'])
    def delete(self):
        rep = reqparse.RequestParser()
        rep.add_argument('user_name', type=validator.validate_max50)
        rep.add_argument('time_from', type=validator.validate_datetime)
        rep.add_argument('time_to', type=validator.validate_datetime)
        rep.add_argument('device_ip', type=str)
        rep.add_argument('user_ip', type=str)
        rep.add_argument('detail', type=str)
        rep.add_argument('user_log_not_selected_ids', type=int, action='append')
        query_args = rep.parse_args()

        query = UserOperation.query.order_by(desc(UserOperation.created_at))

        user_name_text = ''
        user_ip = ''
        time_text = ''
        device_ip = ''
        detail_text = ''

        if query_args.user_name:
            query = query.join(User, User.id == UserOperation.user_id) \
                .filter(User.user_name.ilike('%{}%'.format(query_args.user_name).replace('\b', '').replace('_', '\_')))
            user_name_text = ',1: %s' % query_args.user_name

        if query_args.time_from and query_args.time_to:
            query = query.filter(UserOperation.created_at >= query_args.time_from)
            query = query.filter(UserOperation.created_at <= query_args.time_to)
            time_text = ',1: %s -- %s' % (query_args.time_from.strftime("%Y-%m-%d  %H:%M:%S"),
                                           query_args.time_to.strftime("%Y-%m-%d  %H:%M:%S"))

        if query_args.device_ip:
            query = query.filter \
                    (UserOperation.device_ip.ilike(
                    '%{}%'.format(query_args.device_ip).replace('\b', '').replace('_', '\_')))
            device_ip = ',ii: %s' % query_args.device_ip

        if query_args.user_ip:
            query = query.filter \
                (UserOperation.user_ip.ilike('%{}%'.format(query_args.user_ip).replace('\b', '').replace('_', '\_')))
            user_ip = ',ii: %s' % query_args.user_ip

        if query_args.detail:
            query = query.filter \
                (UserOperation.memo.ilike('%{}%'.format(query_args.detail).replace('\b', '').replace('_', '\_')))
            detail_text = ',ii: %s' % query_args.detail

        if query_args.user_log_not_selected_ids:
            for user_log in query:
                if user_log.id not in query_args.user_log_not_selected_ids:
                    user_log.delete(commit=False)
        else:
            for user_log in query:
                user_log.delete(commit=False)

        loop_delete_log_table(SystemLog, False)
        SystemLog.create(
            commit=False,
            memo='ii' % (g.current_user.user_name, get_user_ip())
                 + user_name_text + user_ip + time_text + device_ip + detail_text,
        )
        db.session.commit()
        return success_response()


class UserOperationLogs(RestfulBase):

    @login_required
    @role_required(ROLES['ADMINISTRATOR'], ROLES['COMPTROLLER'])
    def delete(self):
        req = reqparse.RequestParser()
        req.add_argument('user_log_ids', type=int, action='append')
        args = req.parse_args()

        if not args.user_log_ids:
            raise_400_response(message='ii')

        user_log_ids = list(set(args.user_log_ids))
        if len(user_log_ids) == 1:
            user_log = UserOperation.query.filter_by(id=user_log_ids[0]).first()
            if not user_log:
                raise_404_response(message='1')
            loop_delete_log_table(SystemLog, False)
            SystemLog.create(
                commit=False,
                memo='1: %s' % (
                    g.current_user.user_name, get_user_ip(), user_log.created_at.strftime("%Y-%m-%d  %H:%M:%S"),
                    user_log.user_ip, user_log.device_ip, user_log.memo)
            )
        else:
            loop_delete_log_table(SystemLog, False)
            SystemLog.create(
                commit=False,
                memo='1' % (
                    g.current_user.user_name, get_user_ip(), len(user_log_ids))
            )
        for user_log_id in user_log_ids:
            user_log = UserOperation.query.filter_by(id=user_log_id).first()
            if not user_log:
                raise_404_response(message='1')

            user_log.delete(commit=False)
        db.session.commit()

        return success_response()
