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
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b7 b0)
		(in-tower b7)
		(on b4 b7)
		(in-tower b4)
		(red b1)
		(red b5)
		(blue b4)
		(blue b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b1 b4)) (not (on b5 b4)))))
)