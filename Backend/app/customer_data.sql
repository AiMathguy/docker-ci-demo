

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    country VARCHAR(100),
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMP,
    failed_login_attempts INT NOT NULL DEFAULT 0
);

CREATE TABLE user_activity_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    activity_type VARCHAR(100) NOT NULL,
    activity_time TIMESTAMP NOT NULL DEFAULT NOW(),
    ip_address VARCHAR(100),
    device_info TEXT
);


CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_last_login_at ON users(last_login_at);

CREATE INDEX idx_user_activity_user_id ON user_activity_log(user_id);
CREATE INDEX idx_user_activity_time ON user_activity_log(activity_time);
CREATE INDEX idx_user_activity_type ON user_activity_log(activity_type);

INSERT INTO users (full_name, email, role, status, country, is_verified, created_at, last_login_at, failed_login_attempts)
VALUES
('Thato Bilankulu', 'thato@example.com', 'admin', 'active', 'South Africa', true, NOW() - INTERVAL '20 days', NOW() - INTERVAL '2 hours', 0),
('Jane Smith', 'jane@example.com', 'teacher', 'active', 'South Africa', true, NOW() - INTERVAL '14 days', NOW() - INTERVAL '1 day', 1),
('Neo Mokoena', 'neo@example.com', 'student', 'suspended', 'South Africa', false, NOW() - INTERVAL '7 days', NOW() - INTERVAL '2 days', 4),
('Sarah Lee', 'sarah@example.com', 'admin', 'active', 'Kenya', true, NOW() - INTERVAL '30 days', NOW() - INTERVAL '35 days', 0),
('Mike Brown', 'mike@example.com', 'user', 'inactive', 'Botswana', false, NOW() - INTERVAL '40 days', NOW() - INTERVAL '48 days', 2),
('Alice Dube', 'alice@example.com', 'student', 'active', 'Zimbabwe', true, NOW() - INTERVAL '3 days', NOW() - INTERVAL '3 hours', 0),
('Peter Ndlovu', 'peter@example.com', 'teacher', 'active', 'South Africa', false, NOW() - INTERVAL '2 days', NOW() - INTERVAL '12 days', 5),
('Lebo Khumalo', 'lebo@example.com', 'user', 'inactive', 'South Africa', true, NOW() - INTERVAL '12 days', NOW() - INTERVAL '22 days', 0);

INSERT INTO user_activity_log (user_id, activity_type, activity_time, ip_address, device_info)
SELECT id, 'login', NOW() - INTERVAL '1 day', '192.168.1.10', 'Chrome on Windows'
FROM users
WHERE email IN ('thato@example.com', 'jane@example.com', 'alice@example.com');

INSERT INTO user_activity_log (user_id, activity_type, activity_time, ip_address, device_info)
SELECT id, 'login', NOW() - INTERVAL '2 days', '192.168.1.20', 'Edge on Windows'
FROM users
WHERE email IN ('neo@example.com', 'peter@example.com');

INSERT INTO user_activity_log (user_id, activity_type, activity_time, ip_address, device_info)
SELECT id, 'profile_update', NOW() - INTERVAL '4 hours', '192.168.1.30', 'Firefox on Linux'
FROM users
WHERE email IN ('thato@example.com', 'jane@example.com');

UPDATE users
SET password_hash = 'PASTE_HASH_HERE'
WHERE email = 'thato@example.com';

UPDATE users
SET password_hash = 'PASTE_HASH_HERE'
WHERE email = 'thato@example.com';

UPDATE users
SET password_hash = 'PASTE_HASH_HERE'
WHERE email = 'thato@example.com';


UPDATE users
SET created_at = CASE email
  WHEN 'thato@example.com' THEN NOW() - INTERVAL '30 days'
  WHEN 'jane@example.com' THEN NOW() - INTERVAL '24 days'
  WHEN 'neo@example.com' THEN NOW() - INTERVAL '18 days'
  WHEN 'sarah@example.com' THEN NOW() - INTERVAL '14 days'
  WHEN 'mike@example.com' THEN NOW() - INTERVAL '10 days'
  WHEN 'alice@example.com' THEN NOW() - INTERVAL '6 days'
  WHEN 'peter@example.com' THEN NOW() - INTERVAL '3 days'
  WHEN 'lebo@example.com' THEN NOW() - INTERVAL '1 day'
  ELSE created_at
END;

SELECT * FROM users;

SELECT * FROM ai_reports;

Select * FROM Users;
SELECT * FROM user_activity log;


