(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b1)
		(clear b1)
		(clear b2)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b3 b0)
		(in-tower b3)
		(on b2 b3)
		(in-tower b2)
		(red b1)
		(red b2)
		(blue b4)
		(blue b5)
		(red b6)
		(blue b7)
		(red b9)
		(blue b9)
		(blue b6)
		(red b7)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b1 b0)) (not (on b1 b2)))))
)