section 1-------

1.1 porcentaje de presencia en el dataset
select
    FLAG_DOCUMENT_X,
    count(*) as total,
    100.0 * count(*) / sum(count(*)) over () as percentage
from application_train
group by FLAG_DOCUMENT_X;

1.2 porcentaje de default en la flag / conteo de eventos
select
    target,
    count(*) as total,
    100.0 * count(*) / sum(count(*)) over () as percentage
from application_train
where FLAG_DOCUMENT_X = '1'
group by target;