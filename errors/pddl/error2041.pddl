(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(clear b7)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b8 b0)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(red b2)
		(blue b3)
		(blue b4)
		(blue b5)
		(red b6)
		(blue b7)
		(red b9)
		(red b7)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b1 b0)) (not (on b9 b0)) (not (on b9 b8)) (not (on b9 b7)))))
)