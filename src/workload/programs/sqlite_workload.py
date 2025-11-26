import os
import random
import string
from typing import Dict, List, Optional, Tuple, Any

from ..base_workload_generator import BaseWorkloadGenerator


class SQLiteWorkloadGenerator(BaseWorkloadGenerator):
    """Workload generator for SQLite"""

    def prepare_inputs(self, mode: str, from_db: bool = False, db_query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Prepare SQLite inputs"""
        if from_db and db_query:
            return self.db_manager.get_previous_parameters(f"{self.program_name}-{mode}", db_query, server=False)
        else:
            workloads = []
            for i in range(self.num_data_points):
                max_tables = random.randint(1, 5)
                max_columns = random.randint(2, 10)
                max_rows = random.randint(20, 1000)

                workload = {
                    'max_tables': max_tables,
                    'max_columns': max_columns,
                    'max_rows': max_rows,
                    'workload_id': i,
                    'use_indexes': random.random() > 0.3,
                    'use_views': random.random() > 0.5,
                    'use_triggers': random.random() > 0.6,
                    'use_transactions': random.random() > 0.4,
                    'use_ctes': random.random() > 0.7,
                    'use_window_functions': random.random() > 0.8,
                    'use_temp_tables': random.random() > 0.6,
                    'use_constraints': random.random() > 0.4,
                    'use_json': random.random() > 0.7,
                    'use_virtual_tables': random.random() > 0.85,
                    'complexity_level': random.choice(['low', 'medium', 'high'])
                }

                if workload not in workloads:
                    workloads.append(workload)
                else:
                    i -= 1

            return workloads

    def prepare_commands(self, input_data: Dict[str, Any], build_type: str,
                         custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare SQLite commands"""
        import tempfile

        max_tables = input_data.get('max_tables', 3)
        max_columns = input_data.get('max_columns', 5)
        max_rows = input_data.get('max_rows', 500)
        workload_id = input_data.get('workload_id', 0)

        use_indexes = input_data.get('use_indexes', False)
        use_views = input_data.get('use_views', False)
        use_triggers = input_data.get('use_triggers', False)
        use_transactions = input_data.get('use_transactions', False)
        use_ctes = input_data.get('use_ctes', False)
        use_window_functions = input_data.get('use_window_functions', False)
        use_temp_tables = input_data.get('use_temp_tables', False)
        use_constraints = input_data.get('use_constraints', False)
        use_json = input_data.get('use_json', False)
        use_virtual_tables = input_data.get('use_virtual_tables', False)
        complexity = input_data.get('complexity_level', 'medium')

        sql_commands = []
        table_names = []
        table_columns = {}

        pragma_options = [
            ("foreign_keys", ["ON", "OFF"]),
            ("journal_mode", ["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"]),
            ("synchronous", ["OFF", "NORMAL", "FULL", "EXTRA"]),
            ("temp_store", ["DEFAULT", "FILE", "MEMORY"]),
            ("cache_size", ["-2000", "-5000", "-10000"]),
            ("recursive_triggers", ["ON", "OFF"]),
            ("case_sensitive_like", ["ON", "OFF"]),
        ]

        num_pragmas = random.randint(1, 4)
        for _ in range(num_pragmas):
            pragma, values = random.choice(pragma_options)
            value = random.choice(values)
            sql_commands.append(f"PRAGMA {pragma} = {value};")

        drop_commands = []
        for t in range(max_tables):
            table_name = f"table_{workload_id}_{t}"
            drop_commands.append(f"DROP TABLE IF EXISTS {table_name};")
            drop_commands.append(f"DROP VIEW IF EXISTS view_{table_name};")
            drop_commands.append(f"DROP INDEX IF EXISTS idx_{table_name};")
            drop_commands.append(f"DROP TRIGGER IF EXISTS trig_{table_name};")

        sql_commands.extend(drop_commands)

        num_tables = random.randint(1, max_tables)

        for t in range(num_tables):
            table_name = f"table_{workload_id}_{t}"
            table_names.append(table_name)

            is_temp = use_temp_tables and random.random() > 0.7
            table_keyword = "TEMP TABLE" if is_temp else "TABLE"

            num_columns = random.randint(2, max_columns)
            columns = []
            column_info = []
            column_constraints = {}

            column_types = ['INTEGER', 'TEXT', 'REAL', 'NUMERIC']

            for c in range(num_columns):
                col_name = f"col_{c}"
                col_type = random.choice(column_types)
                col_def = f"{col_name} {col_type}"
                has_check_constraint = False
                has_not_null = False
                has_unique = False

                if use_constraints:
                    constraints = []
                    if random.random() > 0.8:
                        constraints.append("NOT NULL")
                        has_not_null = True
                    if random.random() > 0.9:
                        constraints.append("UNIQUE")
                        has_unique = True
                    if random.random() > 0.85:
                        if col_type in ['INTEGER', 'REAL', 'NUMERIC']:
                            constraints.append(f"CHECK({col_name} >= 0)")
                            has_check_constraint = True
                    if random.random() > 0.8:
                        if col_type == 'INTEGER':
                            default_val = random.randint(0, 100) if has_check_constraint else random.randint(-100, 100)
                            constraints.append(f"DEFAULT {default_val}")
                        elif col_type == 'TEXT':
                            constraints.append(f"DEFAULT 'default_value'")
                    if random.random() > 0.9 and col_type == 'TEXT':
                        constraints.append("COLLATE NOCASE")

                    if constraints:
                        col_def += " " + " ".join(constraints)

                columns.append(col_def)
                column_info.append((col_name, col_type))
                column_constraints[col_name] = {
                    'has_check': has_check_constraint,
                    'not_null': has_not_null,
                    'has_unique': has_unique
                }

            table_constraints = []
            composite_unique_cols = []
            if use_constraints and len(column_info) > 1 and random.random() > 0.7:
                cols = random.sample([c[0] for c in column_info], min(2, len(column_info)))
                table_constraints.append(f"UNIQUE({', '.join(cols)})")
                composite_unique_cols = cols

            table_columns[table_name] = (column_info, column_constraints, composite_unique_cols)

            create_cmd = f"CREATE {table_keyword} IF NOT EXISTS {table_name} ("
            create_cmd += "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            create_cmd += ", ".join(columns)
            if table_constraints:
                create_cmd += ", " + ", ".join(table_constraints)
            create_cmd += ");"
            sql_commands.append(create_cmd)

        unique_values_tracker = {}
        composite_unique_tracker = {}

        transaction_active = False
        for t in range(num_tables):
            table_name = table_names[t]
            column_data = table_columns[table_name]

            if isinstance(column_data, tuple):
                if len(column_data) == 3:
                    column_info, column_constraints, composite_unique_cols = column_data
                else:
                    column_info, column_constraints = column_data
                    composite_unique_cols = []
            else:
                column_info = column_data
                column_constraints = {}
                composite_unique_cols = []

            for col_name, _ in column_info:
                if column_constraints.get(col_name, {}).get('has_unique', False):
                    unique_values_tracker[(table_name, col_name)] = set()

            if composite_unique_cols:
                composite_unique_tracker[table_name] = set()

            num_rows = random.randint(10, max_rows)

            if use_transactions and random.random() > 0.5 and not transaction_active:
                sql_commands.append("BEGIN TRANSACTION;")
                transaction_active = True

            for r in range(num_rows):
                col_names = [col[0] for col in column_info]
                values = []

                max_retries = 100
                retry_count = 0
                row_values = {}

                while retry_count < max_retries:
                    row_values = {}
                    for col_name, col_type in column_info:
                        has_check = column_constraints.get(col_name, {}).get('has_check', False)
                        not_null = column_constraints.get(col_name, {}).get('not_null', False)
                        has_unique = column_constraints.get(col_name, {}).get('has_unique', False)

                        value = None

                        if random.random() < 0.05 and not not_null and not has_unique:
                            value = "NULL"
                        elif col_type == 'INTEGER':
                            if has_check:
                                raw_val = random.randint(0, 1000000)
                            else:
                                raw_val = random.randint(-1000000, 1000000)

                            if has_unique:
                                tracker_key = (table_name, col_name)
                                attempts = 0
                                while raw_val in unique_values_tracker[tracker_key] and attempts < 50:
                                    if has_check:
                                        raw_val = random.randint(0, 1000000)
                                    else:
                                        raw_val = random.randint(-1000000, 1000000)
                                    attempts += 1

                                if raw_val in unique_values_tracker[tracker_key]:
                                    raw_val = r + (workload_id * 10000000)

                                unique_values_tracker[tracker_key].add(raw_val)

                            value = str(raw_val)

                        elif col_type == 'REAL' or col_type == 'NUMERIC':
                            if has_check:
                                raw_val = random.uniform(0.0, 10000.0)
                            else:
                                raw_val = random.uniform(-10000.0, 10000.0)

                            if has_unique:
                                tracker_key = (table_name, col_name)
                                raw_val = raw_val + (r * 0.0001) + (workload_id * 0.1)
                                unique_values_tracker[tracker_key].add(raw_val)

                            value = str(raw_val)

                        elif col_type == 'BLOB':
                            blob_size = random.randint(4, 20)
                            blob_data = ''.join(random.choices('0123456789ABCDEF', k=blob_size))

                            if has_unique:
                                tracker_key = (table_name, col_name)
                                blob_data = f"{r:04X}{workload_id:04X}{blob_data}"
                                unique_values_tracker[tracker_key].add(blob_data)

                            value = f"X'{blob_data}'"

                        else:
                            if has_unique or col_name in composite_unique_cols:
                                unique_id = f"{table_name}_{col_name}_{workload_id}_{r}_{retry_count}"
                                text_patterns = [
                                    lambda: f"unique_{unique_id}_{random.randint(1, 1000000)}",
                                    lambda: f"user_{unique_id}@example.com",
                                    lambda: f"data_{unique_id}_{random.choice(['a', 'b', 'c'])}",
                                ]
                                text_gen = random.choice(text_patterns)
                                text_val = text_gen()

                                if has_unique:
                                    tracker_key = (table_name, col_name)
                                    attempts = 0
                                    while text_val in unique_values_tracker[tracker_key] and attempts < 50:
                                        text_gen = random.choice(text_patterns)
                                        text_val = text_gen()
                                        attempts += 1

                                    if text_val in unique_values_tracker[tracker_key]:
                                        text_val = f"unique_{table_name}_{col_name}_{workload_id}_{r}_{random.randint(1000000, 9999999)}"

                                    unique_values_tracker[tracker_key].add(text_val)
                            else:   
                                text_patterns = [
                                    lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 30))),
                                    lambda: f"user_{random.randint(1, 100000)}@example.com",
                                    lambda: f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}_{r}",
                                    lambda: f"{{\"key\": \"value_{random.randint(1, 10000)}_{r}\"}}",
                                    lambda: f"{random.choice(['quoted', 'special', 'unicode'])}_{r}_{random.randint(1, 1000)}",
                                    lambda: f"text_{workload_id}_{t}_{r}_{random.randint(1, 10000)}",
                                ]
                                text_gen = random.choice(text_patterns)
                                text_val = text_gen()

                            text_val = text_val.replace("'", "''")
                            value = f"'{text_val}'"

                        row_values[col_name] = value

                    if composite_unique_cols:
                        composite_key = tuple(row_values[col] for col in composite_unique_cols)

                        if composite_key in composite_unique_tracker[table_name]:
                            retry_count += 1
                            continue
                        else:
                            composite_unique_tracker[table_name].add(composite_key)

                    break

                for col_name in col_names:
                    values.append(row_values[col_name])

                if random.random() > 0.7:
                    insert_type = random.choice(["INSERT OR REPLACE", "INSERT OR IGNORE", "REPLACE"])
                    insert_cmd = f"{insert_type} INTO {table_name} (id, {', '.join(col_names)}) VALUES ({r}, {', '.join(values)});"
                else:
                    insert_cmd = f"INSERT INTO {table_name} (id, {', '.join(col_names)}) VALUES ({r}, {', '.join(values)});"

                sql_commands.append(insert_cmd)

            if transaction_active and random.random() > 0.5:
                sql_commands.append("COMMIT;")
                transaction_active = False

        if transaction_active:
            sql_commands.append("COMMIT;")
            transaction_active = False

        if use_indexes and table_names:
            num_indexes = random.randint(1, min(3, len(table_names)))
            for _ in range(num_indexes):
                table = random.choice(table_names)
                column_data = table_columns.get(table, ([], {}, []))
                cols = column_data[0] if isinstance(column_data, tuple) else column_data
                if cols:
                    index_cols = random.sample([c[0] for c in cols], min(random.randint(1, 3), len(cols)))
                    index_name = f"idx_{table}_{random.randint(1, 1000)}"

                    index_type = random.choice(["", "UNIQUE"])
                    where_clause = ""
                    if random.random() > 0.7:
                        col = random.choice(index_cols)
                        where_clause = f" WHERE {col} IS NOT NULL"

                    sql_commands.append(f"CREATE {index_type} INDEX IF NOT EXISTS {index_name} ON {table} ({', '.join(index_cols)}){where_clause};")

        if use_views and len(table_names) > 0:
            num_views = random.randint(1, min(2, len(table_names)))
            for v in range(num_views):
                view_name = f"view_{workload_id}_{v}"
                table = random.choice(table_names)
                column_data = table_columns.get(table, ([], {}, []))
                cols = column_data[0] if isinstance(column_data, tuple) else column_data

                if cols:
                    if len(table_names) > 1 and random.random() > 0.5:
                        t1 = random.choice(table_names)
                        t2 = random.choice([t for t in table_names if t != t1])
                        sql_commands.append(f"""
                        CREATE VIEW IF NOT EXISTS {view_name} AS
                        SELECT t1.id as id1, t2.id as id2, t1.col_0 as data
                        FROM {t1} t1
                        LEFT JOIN {t2} t2 ON t1.id = t2.id;
                        """)
                    else:
                        col = random.choice([c[0] for c in cols])
                        sql_commands.append(f"""
                        CREATE VIEW IF NOT EXISTS {view_name} AS
                        SELECT {col}, COUNT(*) as cnt
                        FROM {table}
                        GROUP BY {col};
                        """)

        if use_triggers and table_names:
            num_triggers = random.randint(1, min(2, len(table_names)))
            for tr in range(num_triggers):
                table = random.choice(table_names)
                trigger_name = f"trig_{workload_id}_{tr}"
                trigger_time = random.choice(["BEFORE", "AFTER"])
                trigger_event = random.choice(["INSERT", "UPDATE", "DELETE"])

                sql_commands.append(f"""
                CREATE TRIGGER IF NOT EXISTS {trigger_name}
                {trigger_time} {trigger_event} ON {table}
                BEGIN
                    SELECT RAISE(IGNORE);
                END;
                """)

        num_queries = random.randint(20, 50) if complexity in ['high', 'extreme'] else random.randint(10, 30)

        for _ in range(num_queries):
            if not table_names:
                continue

            if complexity == 'extreme':
                query_weights = [20, 15, 20, 15, 10, 10, 5, 5]
                query_types = ['SIMPLE', 'AGGREGATE', 'JOIN', 'SUBQUERY', 'COMPLEX', 'CTE', 'WINDOW', 'SET_OPS']
            elif complexity == 'high':
                query_weights = [30, 20, 15, 15, 10, 5, 3, 2]
                query_types = ['SIMPLE', 'AGGREGATE', 'JOIN', 'SUBQUERY', 'COMPLEX', 'CTE', 'WINDOW', 'SET_OPS']
            else:
                query_weights = [50, 20, 15, 10, 5]
                query_types = ['SIMPLE', 'AGGREGATE', 'JOIN', 'SUBQUERY', 'COMPLEX']

            query_type = random.choices(query_types, weights=query_weights[:len(query_types)], k=1)[0]

            if query_type == 'SIMPLE':
                table = random.choice(table_names)
                operation = random.choice(['SELECT', 'UPDATE', 'DELETE'])

                if operation == 'SELECT':
                    select_patterns = [
                        f"SELECT * FROM {table} WHERE id > {random.randint(0, 100)} LIMIT {random.randint(1, 50)};",
                        f"SELECT DISTINCT col_0 FROM {table};",
                        f"SELECT * FROM {table} WHERE id BETWEEN {random.randint(1, 50)} AND {random.randint(51, 100)};",
                        f"SELECT * FROM {table} WHERE col_0 LIKE 'user%';",
                        f"SELECT * FROM {table} WHERE id IN ({', '.join(str(random.randint(1, 100)) for _ in range(random.randint(3, 10)))});",
                        f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {random.randint(5, 20)};",
                    ]
                    sql_commands.append(random.choice(select_patterns))

                elif operation == 'UPDATE':
                    sql_commands.append(f"UPDATE {table} SET col_0 = col_0 || '_updated' WHERE id % {random.randint(2, 10)} = 0;")
                else:
                    sql_commands.append(f"DELETE FROM {table} WHERE id % {random.randint(5, 20)} = 0;")

            elif query_type == 'AGGREGATE':
                table = random.choice(table_names)
                column_data = table_columns.get(table, ([], {}, []))
                column_info = column_data[0] if isinstance(column_data, tuple) else column_data

                if column_info:
                    agg_patterns = [
                        f"SELECT COUNT(*), AVG(id), MIN(id), MAX(id) FROM {table};",
                        f"SELECT col_0, COUNT(*) FROM {table} GROUP BY col_0 HAVING COUNT(*) > 1;",
                        f"SELECT COUNT(DISTINCT col_0) FROM {table};",
                        f"SELECT GROUP_CONCAT(col_0) FROM {table} WHERE id < 10;",
                        f"SELECT col_0, SUM(id) OVER (ORDER BY id) FROM {table} LIMIT 20;",
                    ]
                    sql_commands.append(random.choice(agg_patterns))

            elif query_type == 'JOIN':
                if len(table_names) > 1:
                    t1 = random.choice(table_names)
                    t2 = random.choice([t for t in table_names if t != t1])

                    join_patterns = [
                        f"SELECT * FROM {t1} NATURAL JOIN {t2};",
                        f"SELECT * FROM {t1} CROSS JOIN {t2} LIMIT 10;",
                        f"SELECT * FROM {t1} t1 INNER JOIN {t2} t2 USING (id);",
                        f"SELECT * FROM {t1} t1 LEFT OUTER JOIN {t2} t2 ON t1.id = t2.id;",
                        f"SELECT * FROM {t1} t1 FULL OUTER JOIN {t2} t2 ON t1.id = t2.id;",
                    ]
                    sql_commands.append(random.choice(join_patterns))

            elif query_type == 'SUBQUERY':
                if table_names:
                    table = random.choice(table_names)

                    subquery_patterns = [
                        f"SELECT * FROM {table} WHERE id = (SELECT MAX(id) FROM {table});",
                        f"SELECT * FROM {table} WHERE EXISTS (SELECT 1 FROM {table} t2 WHERE t2.id > {table}.id);",
                        f"SELECT * FROM {table} WHERE id NOT IN (SELECT id FROM {table} WHERE id < 10);",
                        f"SELECT (SELECT COUNT(*) FROM {table}) as total_count;",
                        f"SELECT * FROM (SELECT * FROM {table} ORDER BY id DESC LIMIT 10) ORDER BY id ASC;",
                    ]
                    sql_commands.append(random.choice(subquery_patterns))

            elif query_type == 'CTE' and use_ctes:
                if table_names:
                    table = random.choice(table_names)

                    cte_patterns = [
                        f"""WITH cte AS (SELECT * FROM {table} WHERE id < 50)
                            SELECT COUNT(*) FROM cte;""",
                        f"""WITH RECURSIVE cnt(x) AS (
                            VALUES(1)
                            UNION ALL
                            SELECT x+1 FROM cnt WHERE x < 10
                        ) SELECT x FROM cnt;""",
                        f"""WITH 
                            t1 AS (SELECT * FROM {table} WHERE id < 20),
                            t2 AS (SELECT * FROM {table} WHERE id >= 20 AND id < 40)
                            SELECT * FROM t1 UNION ALL SELECT * FROM t2;""",
                    ]
                    sql_commands.append(random.choice(cte_patterns))

            elif query_type == 'WINDOW' and use_window_functions:
                if table_names:
                    table = random.choice(table_names)

                    window_patterns = [
                        f"SELECT id, ROW_NUMBER() OVER (ORDER BY id) as rn FROM {table} LIMIT 20;",
                        f"SELECT id, RANK() OVER (ORDER BY col_0) as rnk FROM {table} LIMIT 20;",
                        f"SELECT id, LAG(id, 1) OVER (ORDER BY id) as prev_id FROM {table} LIMIT 20;",
                        f"SELECT id, FIRST_VALUE(col_0) OVER (PARTITION BY id % 10 ORDER BY id) FROM {table} LIMIT 20;",
                        f"SELECT id, NTILE(4) OVER (ORDER BY id) as quartile FROM {table};",
                    ]
                    sql_commands.append(random.choice(window_patterns))

            elif query_type == 'SET_OPS':
                if len(table_names) > 1:
                    t1 = random.choice(table_names)
                    t2 = random.choice([t for t in table_names if t != t1])

                    set_patterns = [
                        f"SELECT id FROM {t1} UNION SELECT id FROM {t2};",
                        f"SELECT id FROM {t1} UNION ALL SELECT id FROM {t2};",
                        f"SELECT id FROM {t1} INTERSECT SELECT id FROM {t2};",
                        f"SELECT id FROM {t1} EXCEPT SELECT id FROM {t2};",
                    ]
                    sql_commands.append(random.choice(set_patterns))

            elif query_type == 'COMPLEX':
                if table_names:
                    table = random.choice(table_names)

                    complex_patterns = [
                        f"""SELECT 
                            id,
                            CASE 
                                WHEN id < 10 THEN 'Low'
                                WHEN id < 50 THEN 'Medium'
                                ELSE 'High'
                            END as category,
                            LENGTH(col_0) as str_len,
                            SUBSTR(col_0, 1, 5) as prefix,
                            UPPER(col_0) as upper_col
                        FROM {table}
                        WHERE id % 3 = 0
                        ORDER BY LENGTH(col_0) DESC, id ASC
                        LIMIT 15;""",

                        f"""SELECT 
                            strftime('%Y-%m', 'now') as month,
                            COUNT(*) as cnt,
                            GROUP_CONCAT(DISTINCT col_0) as grouped_values
                        FROM {table}
                        WHERE julianday('now') - julianday('2024-01-01') > id
                        GROUP BY id % 5
                        HAVING COUNT(*) > 0;""",
                    ]
                    sql_commands.append(random.choice(complex_patterns))

        if use_json and table_names:
            table = random.choice(table_names)
            json_patterns = [
                f"SELECT json_extract(col_0, '$.key') FROM {table} WHERE json_valid(col_0) LIMIT 10;",
                f"SELECT json_array(id, col_0) FROM {table} LIMIT 10;",
                f"SELECT json_object('id', id, 'data', col_0) FROM {table} LIMIT 10;",
            ]
            for pattern in random.sample(json_patterns, min(2, len(json_patterns))):
                sql_commands.append(pattern)

        if random.random() > 0.8:
            sql_commands.append("ANALYZE;")
        if random.random() > 0.9:
            sql_commands.append("VACUUM;")

        if use_transactions and random.random() > 0.7 and table_names:
            sql_commands.extend([
                "BEGIN TRANSACTION;",
                "SAVEPOINT sp1;",
                f"DELETE FROM {random.choice(table_names)} WHERE id > 1000;",
                "ROLLBACK TO sp1;",
                "COMMIT;",
            ])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as temp_file:
            temp_file.write("\n".join(sql_commands))
            temp_file_path = temp_file.name

        db_file = os.path.join(self.program_build_dir, f"random_db_{workload_id}.db")
        if os.path.exists(db_file):
            os.remove(db_file)

        program_command = [
            './sqlite3',
            db_file,
            '<',
            temp_file_path
        ]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'sql_file': temp_file_path,
            'db_file': db_file,
            'num_tables': num_tables,
            'num_commands': len(sql_commands),
            'workload_id': workload_id,
            'features_used': {
                'indexes': use_indexes,
                'views': use_views,
                'triggers': use_triggers,
                'transactions': use_transactions,
                'ctes': use_ctes,
                'window_functions': use_window_functions,
                'temp_tables': use_temp_tables,
                'constraints': use_constraints,
                'json': use_json,
                'virtual_tables': use_virtual_tables,
            },
            'complexity': complexity
        }

        input_data['parameters'] = parameters

        return program_command, parameters

    def _pre_iteration(self, input_data: Any) -> None:
        """Perform setup before each iteration"""
        parameters = input_data.get('parameters', {})
        db_file = parameters.get('db_file')
        if db_file and os.path.exists(db_file):
            backup_file = f"{db_file}.bak"
            try:
                if os.path.exists(backup_file):
                    os.unlink(backup_file)
                os.rename(db_file, backup_file)
            except Exception as e:
                self.logger.warning(f"Failed to backup existing database: {e}")

    def _post_iteration(self, input_data: Any) -> None:
        """Perform actions after each iteration"""
        parameters = input_data.get('parameters', {})
        db_file = parameters.get('db_file')
        backup_file = f"{db_file}.bak" if db_file else None
        if backup_file and os.path.exists(backup_file):
            try:
                if os.path.exists(db_file):
                    os.unlink(db_file)
                os.rename(backup_file, db_file)
            except Exception as e:
                self.logger.warning(f"Failed to restore database from backup: {e}")

    def _cleanup_after_input(self, input_data: Dict[str, Any]) -> None:
        """Clean up the temporary SQL file and database after processing"""

        parameters = input_data.get('parameters', {})
        sql_file = parameters.get('sql_file')
        db_file = parameters.get('db_file')

        try:
            if sql_file and os.path.exists(sql_file):
                os.unlink(sql_file)

            if db_file and os.path.exists(db_file):
                os.unlink(db_file)
        except Exception as e:
            self.logger.warning(f"Failed to clean up files: {e}")
